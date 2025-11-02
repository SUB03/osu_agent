import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as v2
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from python_osu_parser import *
import pygetwindow as gw
from OsuEnv import OsuEnv
from collections import deque
from matplotlib import pyplot as plt
import time

import utils
HOST = "127.0.0.1"
PORT = 13000

# Hyperparameters
N_EPISODES = 10000
MAX_STEPS = 5000
EPOCHS = 10

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class OsuAI(nn.Module):
    def __init__(self, device):
        super(OsuAI, self).__init__()
        self.device = device

        self.transforms = v2.Compose([
            v2.Resize((84, 84)),
        ])

        self.preprocess = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4),
            nn.BatchNorm2d(32),
            nn.SiLU(),

            nn.Conv2d(32, 64, 4, 2),
            nn.BatchNorm2d(64),
            nn.SiLU(),

            nn.Conv2d(64, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Flatten(),
        )

        # self.preprocess = nn.Sequential(
        #     nn.Conv2d(4, 64, 5, 3),
        #     nn.ReLU(),

        #     nn.Conv2d(64, 128, 5, 2),
        #     nn.ReLU(),

        #     nn.Conv2d(128, 128, 3, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 64, 3, 1),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     layer_init(nn.Linear(3136, 2048)),
        #     nn.ReLU(),
        # )

        # feature detection
        with torch.no_grad():
            dummy = torch.zeros(1, 4, 84, 84)
            features_dim = self.preprocess(dummy).size(1)

        self.continuous_mean = nn.Sequential(
            layer_init(nn.Linear(features_dim, 512)),
            nn.SiLU(),
            layer_init(nn.Linear(512, 512)),
            nn.SiLU(),
            layer_init(nn.Linear(512, 2), std=0.01),
        )
        self.discrete = nn.Sequential(
            layer_init(nn.Linear(features_dim, 512)),
            nn.SiLU(),
            layer_init(nn.Linear(512, 512)),
            nn.SiLU(),
            layer_init(nn.Linear(512, 3), std=0.01),
        )
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(features_dim, 512)),
            nn.SiLU(),
            layer_init(nn.Linear(512, 512)),
            nn.SiLU(),
            layer_init(nn.Linear(512, 1), std=1.),
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, 2))
        self.actor_optimizer = optim.Adam(
            list(self.preprocess.parameters()) +
            list(self.continuous_mean.parameters()) +
            list(self.discrete.parameters()) +
            [self.actor_logstd],
            lr=3e-4,
            betas=(0.999, 0.999)
        )
        self.critic_optimizer = optim.Adam(
            list(self.preprocess.parameters()) +\
            list(self.critic.parameters()),
            lr=1e-3,
            betas=(0.999, 0.999))

        #self.optimizer = optim.Adam(self.parameters(), 3e-4)
        self.to(device)

    def transformImage(self, state):
        return self.transforms(state).to(self.device)

    def save_model(self):
        torch.save(self.state_dict(), "tmp/osuAI_v2.pt")
    
    def load_model(self):
        self.load_state_dict(torch.load("tmp/osuAI_v2.pt"))

    def get_value(self, x):
        x = self.preprocess(x)
        return self.critic(x)

    def choose_action(self, x, action=None):
        x = self.preprocess(x)
        continuous_mean = self.continuous_mean(x)
        discrete_logits = self.discrete(x)
        action_logstd = self.actor_logstd.expand_as(continuous_mean)
        action_std = torch.exp(action_logstd)
        continuous_probs = Normal(continuous_mean, action_std)
        discrete_probs = Categorical(logits=discrete_logits)
        continuous_action = continuous_probs.rsample()
        discrete_action = discrete_probs.sample()
    
        total_prob = continuous_probs.log_prob(continuous_action).sum(-1)\
            + discrete_probs.log_prob(discrete_action.long())
        
        # action = torch.cat([continuous_action, discrete_action.unsqueeze(-1)], dim=-1).squeeze(0)
        total_entropy = continuous_probs.entropy().sum(-1)\
                + discrete_probs.entropy()
        return continuous_action, discrete_action, total_prob, total_entropy

prefix = "osu!"

utils.wait_for_game_to_launch((HOST, PORT))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
osuEnv = OsuEnv((HOST, PORT))
osuai = OsuAI(device)
gamma = 0.99
value_coef = 0.5
clip_eps = 0.2
entropy_coef = 0.01
replay_buffer = deque(maxlen=8)
mb_size = 4
batch_size = 128
osuai.load_model()

# for plotting
entropies = []

# Activate osu! window
osu = gw.getWindowsWithTitle(prefix)[0]
if osu.isMinimized:
    osu.restore()
osu.activate()
time.sleep(0.5)

osuai.eval()
for episode in range(N_EPISODES):
    state = osuEnv.reset()
    state_tensor = osuai.transformImage(torch.as_tensor(state, dtype=torch.float32, device=device))

    states = []
    discrete_actions = []
    continuous_actions = []
    rewards = []
    logprobs = []
    next_states = []

    batch_entropies = []
    
    # Run episode
    done = False
    t = 0
    while not done:
        states.append(state_tensor)
        with torch.no_grad():
            continuous_action, discrete_action, log_prob, _ = osuai.choose_action(state_tensor.unsqueeze(0))
            state, reward, done = osuEnv.step(continuous_action.to('cpu').squeeze(), discrete_action.to('cpu').squeeze())

        continuous_actions.append(continuous_action)
        discrete_actions.append(discrete_action)
        rewards.append(reward)
        logprobs.append(log_prob.squeeze())

        # Prepare next state
        state_tensor = osuai.transformImage(torch.as_tensor(state, dtype=torch.float32, device=device))

        t+=1

    trajectory = osuEnv.get_score()

    replay_buffer.append({
        "states": torch.stack(states),
        "discrete_actions": torch.cat(discrete_actions),
        "continuous_actions": torch.cat(continuous_actions),
        "rewards": rewards,
        "logprobs": torch.stack(logprobs),
        "trajectories": trajectory
    })
    
    # Train after each map as soon as replay buffer filled enough
    num_samples = len(replay_buffer) 
    if num_samples >= mb_size:
        osuai.train()
        # find best trajectory
        best_trajidx = 0
        for i in range(num_samples):
            if replay_buffer[best_trajidx]["trajectories"] < replay_buffer[i]["trajectories"]:
                best_trajidx = i
        
        mb_states = []
        mb_continuous_actions = []
        mb_discrete_actions = []
        mb_rewards = []
        mb_logprobs = []
        mb_indices = np.random.choice([i for i in range(num_samples) if i != best_trajidx], mb_size-1, False)
        mb_indices = np.concatenate(([best_trajidx], mb_indices))
        for i in mb_indices:
            mb_states.append(replay_buffer[i]["states"])
            mb_continuous_actions.append(replay_buffer[i]["continuous_actions"])
            mb_discrete_actions.append(replay_buffer[i]["discrete_actions"])
            mb_rewards.append(replay_buffer[i]["rewards"])
            mb_logprobs.append(replay_buffer[i]["logprobs"])
        
        returns = []
        for traj_rewards in mb_rewards:
            T = len(traj_rewards)
            G = torch.zeros(T, device=device)

            G[-1] = traj_rewards[-1]
            for t in range(T-2, -1, -1):
                G[t] = traj_rewards[t] + gamma * G[t+1]
            returns.append(G)
        advantages = []
        for i, traj_states in enumerate(mb_states):
            with torch.no_grad():
                values = osuai.get_value(traj_states)
                advantages.append(returns[i] - values.squeeze())
        
        flat_states = torch.cat(mb_states)
        flat_continuous_actions = torch.cat(mb_continuous_actions)
        flat_discrete_actions = torch.cat(mb_discrete_actions)
        flat_logprobs = torch.cat(mb_logprobs)
        flat_returns = torch.cat(returns)
        flat_advantages = torch.cat(advantages)

        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
            
        total_transitions = len(flat_states)
        num_batches = int(np.ceil(total_transitions / batch_size))
        for epoch in range(10):
            indices = torch.randperm(total_transitions, device=device)
            for i in range(num_batches):
                start = i * batch_size
                end = start + batch_size
                idx = indices[start:end]

                states_mb = flat_states[idx]
                continuous_actions_mb = flat_continuous_actions[idx]
                discrete_actions_mb = flat_discrete_actions[idx]
                logprobs_mb = flat_logprobs[idx]
                returns_mb = flat_returns[idx]
                advantages_mb = flat_advantages[idx]

                # policy loss
                features = osuai.preprocess(states_mb)

                continuous_mean = osuai.continuous_mean(features)
                discrete_logits = osuai.discrete(features)

                action_logstd = osuai.actor_logstd.expand_as(continuous_mean)
                action_std = torch.exp(action_logstd)

                continuous_probs = Normal(continuous_mean, action_std)
                discrete_probs = Categorical(logits=discrete_logits)
                new_logprobs = continuous_probs.log_prob(continuous_actions_mb).sum(-1) \
                    + discrete_probs.log_prob(discrete_actions_mb.long())
                
                entropy = continuous_probs.entropy().sum(-1) + discrete_probs.entropy()

                ratio = (new_logprobs - logprobs_mb).exp()
                pg_loss1 = ratio * advantages_mb
                pg_loss2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages_mb
                pg_loss = -torch.min(pg_loss1, pg_loss2).mean()

                osuai.actor_optimizer.zero_grad()
                (pg_loss - entropy_coef * entropy.mean()).backward()
                osuai.actor_optimizer.step()

                new_values = osuai.get_value(states_mb).squeeze()
                value_loss = torch.mean((new_values - returns_mb) ** 2)

                osuai.critic_optimizer.zero_grad()
                value_loss.backward()
                osuai.critic_optimizer.step()

                batch_entropies.append(entropy.mean().item())
                
        if (episode+1) % 32 == 0:
            osuai.save_model()
            print("model saved")
        
        entropies.append(np.mean(batch_entropies))

    print(f"episode: {episode+1}/{N_EPISODES}, reward: {sum(rewards)}, trajectory: {trajectory}, t:{t} actor_logstd: {osuai.actor_logstd.data.cpu().numpy()},\
        mean_entropy: {np.mean(batch_entropies) if len(batch_entropies) != 0 else 'no training'}")
    
print("Training completed!")
eps = list(range(1, len(entropies) + 1))
plt.plot(entropies, eps, marker='o', linestyle='-', color='b')

# Add labels and title
plt.ylabel('Entropy')
plt.xlabel('Episode')
plt.title('Entropy over Episodes')

# Show the plot
plt.grid(True)
plot_filename = f'entropy_plots/entropy.png'
plt.savefig(plot_filename)
plt.close()