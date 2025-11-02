import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import OsuEnv

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
    
    
class Agent(nn.Module):
    def __init__(self, gamma=0.99, alpha=2.5e-4, epsilon=3e-5, gae_lambda=0.95,\
            policy_clip=0.2, device='cpu'):
        super(Agent, self).__init__()

        self.device = device
        self.alpha = alpha
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip

        self.transforms = v2.Compose([
            v2.Resize((80, 80)),
        ])

        self.preprocess = nn.Sequential(
            nn.Conv2d(4, 64, 5, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, 2),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(3136, 2048)),
            nn.ReLU(),
        )

        self.continuous_mean = nn.Sequential(
            layer_init(nn.Linear(2048, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 2), std=0.01),
        )
        self.discrete = nn.Sequential(
            layer_init(nn.Linear(2048, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 3), std=0.01),
        )
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(2048, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.),
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, 2))

        self.optimizer = optim.Adam(self.parameters(), lr=alpha, eps=epsilon)
        self.to(device)
    
    def get_value(self, x):
        features = self.preprocess(x)
        return self.critic(features)
    
    def choose_action(self, x, action=None):
        features = self.preprocess(x)
        discrete_logits = self.discrete(features)
        continuous_mean = self.continuous_mean(features)

        continuous_logstd = self.actor_logstd.expand_as(continuous_mean)
        continuous_std = torch.exp(continuous_logstd)
        continuous_probs = Normal(continuous_mean, continuous_std)
        discrete_probs = Categorical(logits=discrete_logits)

        discrete_action = discrete_probs.sample()
        continuous_action = continuous_probs.sample()
        
        total_logprob = continuous_probs.log_prob(continuous_action).sum(-1)\
            + discrete_probs.log_prob(discrete_action)
        
        total_entropy = continuous_probs.entropy().sum(-1)\
            + discrete_probs.entropy()
        
        return continuous_action, discrete_action, total_logprob, total_entropy, self.critic(x)


if __name__ == "__main__":
    HOST = "127.0.0.1"
    PORT = 13000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 128
    n_episodes = 1000000
    n_updates = n_episodes // batch_size
 
    env = OsuEnv((HOST, PORT))
    agent = Agent(device=device)

    # storage
    states = torch.zeros((batch_size, 4, 80, 80)).to(device)
    discrete_actions = torch.zeros(batch_size).to(device)
    continuous_actions = torch.zeros((batch_size, 3)).to(device)
    logprobs = torch.zeros(batch_size).to(device)
    rewards = torch.zeros(batch_size).to(device)
    dones = torch.zeros(batch_size).to(device)
    values = torch.zeros(batch_size).to(device) 

    global_step = 0
    start_time = time.time()
    state = env.reset()
    next_state = torch.tensor(state).to(device)
    next_done = torch.zeros(1).to(device)

    for update in range(1, n_updates+1):
        agent.eval()

        # lr annealing
        frac = 1.0 - (update - 1.0) / n_updates
        lrnow = frac * agent.alpha
        agent.optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, batch_size):
            global_step += 1
            states[step] = next_state
            dones[step] = next_done

            with torch.no_grad():
                continuous_action, discrete_action, logprob, _, value = agent.choose_action(next_state) 
            discrete_actions[step] = discrete_action
            continuous_actions[step] = continuous_action
            logprobs[step] = logprob
            values[step] = value

            next_state, reward, done, info = env.step(continuous_action.item())
            rewards[step] = torch.tensor(reward).to(device)
            next_state, done = torch.tensor(next_state).to(device), torch.tensor(done).to(device)
        
        # GAE
        with torch.no_grad():
            next_value = agent.get_value(next_state)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(batch_size)):
                if t == batch_size - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t+1]
                    nextvalues = values[t+1]
                delta = rewards[t] + agent.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + agent.gamma * agent.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values 

        # flatten the batch if you more then one env
        
        # training
        agent.train()
        _, newlogprob, entropy, newvalue = agent.choose_action(states, actions)
        ratio = (newlogprob - logprobs).exp()

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - 0.2, 1 + 0.2)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # value loss
        newvalue = newvalue.view(-1)
        v_loss_unclipped = (newvalue - returns) ** 2
        v_clipped = values + torch.clamp(newvalue - values, -0.2, 0.2)
        v_loss_clipped = (v_clipped - returns) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - 0.01 * entropy_loss + v_loss * 0.5

        agent.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
        agent.optimizer.step()