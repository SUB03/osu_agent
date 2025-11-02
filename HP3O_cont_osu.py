import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Independent, Categorical
from ReplayDataset import ReplayDataset
from torch.utils.data import DataLoader
from OsuEnv import OsuEnv
from matplotlib import pyplot as plt
from tqdm import tqdm
import argparse
import time

from RNDModel import RNDModel
from RunningMean import RunningMeanStd
from Trajectory_Replay_Buffer_optimized import TrajectoryBuffer

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)
    return layer

#region Model
class HP3O(nn.Module):
    def __init__(self, device, buffer_size, trajectory_sample_size, obs_space, cont_action_space, disc_action_space):
        super(HP3O, self).__init__()
        self.buffer_size = buffer_size
        self.device = device
        self.n_epochs = 10

        self.gamma = 0.995
        self.gae_lambda = 0.97
        self.clip_range = 0.2
        self.clip_range_vf = 0.2
        self.data_sample_size = 35000
        self.batch_size = 1024
        self.target_kl = None
        self.ent_coef = 0.0004
        self.vf_coef = 0.5
        self.max_grad_norm = 0.8
        self.verbose = 1
        self.lr = 3e-4
        self.use_sde = False
        self.trajectory_sample_size = trajectory_sample_size
        self.trajectory_buffer = TrajectoryBuffer(buffer_size, device, self.gamma, self.gae_lambda)

        self.preprocess = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=5, stride=3),  
            nn.SiLU(),

            nn.Conv2d(64, 128, kernel_size=5, stride=2),  
            nn.SiLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=1),  
            nn.SiLU(),

            nn.Flatten(),
        )
        
        with th.no_grad():
            dummy = th.zeros(1, *obs_space)
            feature_size = self.preprocess(dummy).size(1)

        self.state_encoder = nn.Sequential(
            layer_init(nn.Linear(feature_size+2, 2048)),
            nn.SiLU(),
        )

        self.cont_actor = nn.Sequential(
            layer_init(nn.Linear(2048, 512)),
            nn.SiLU(),
            layer_init(nn.Linear(512, 256)),
            nn.SiLU(),
            layer_init(nn.Linear(256, cont_action_space), std=0.01),
        )

        self.disc_actor = nn.Sequential(
            layer_init(nn.Linear(2048, 512)),
            nn.SiLU(),
            layer_init(nn.Linear(2048, 256)),
            nn.SiLU(),
            layer_init(nn.Linear(256, disc_action_space), std=0.01),
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(2048, 512)),
            nn.SiLU(),
            layer_init(nn.Linear(512, 256)),
            nn.SiLU(),
            layer_init(nn.Linear(256, 1), std=1.),
        )

        self.logstd = nn.Parameter(th.ones(1, cont_action_space) * -1.5)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, betas=(0.999, 0.999))
        self.to(device)
    
    def forward(self, x, cursor):
        x = self.preprocess(x)
        x = th.cat([x, cursor], dim=1)
        x = self.state_encoder(x)
        return x
    
    def choose_action(self, x, cursor):
        x = self.forward(x, cursor)
        mu = self.cont_actor(x)
        std = F.softplus(self.logstd.expand_as(mu)) + 1e-8
        disc_logits = self.disc_actor(x)
        
        cont_probs = Independent(Normal(mu, std), 1)
        disc_probs = Categorical(logits=disc_logits)

        cont_action = cont_probs.rsample()
        disc_action = disc_probs.sample()

        total_logprob = cont_probs.log_prob(cont_action) +\
            disc_probs.log_prob(disc_action)

        value = self.critic(x).detach().squeeze(0)
        return cont_action.detach().squeeze(0), disc_action.detach(), total_logprob.detach(), value
    
    def evaluate_actions(self, obs, cont_action, disc_action, cursor):
        features = self.forward(obs, cursor)
        mu = self.cont_actor(features)
        std = F.softplus(self.logstd.expand_as(mu)) + 1e-8
        disc_logits = self.disc_actor(features)

        cont_probs = Independent(Normal(mu, std), 1)
        disc_probs = Categorical(logits=disc_logits)

        total_probs = cont_probs.log_prob(cont_action) +\
            disc_probs.log_prob(disc_action)
        
        total_entropy = cont_probs.entropy() + disc_probs.entropy()

        return self.critic(features), total_probs, total_entropy
    
    def add(self, obs, cont_action, disc_action, reward, done, value, log_prob, cursor_pos):
        self.trajectory_buffer.add(obs, cont_action, disc_action, reward, done, value, log_prob, cursor_pos)
    
    def save_model(self):
        th.save(self.state_dict(), "tmp/hp3o_online_trained.pt")

    def load_model(self, path="tmp/hp3o_online_trained.pt"):
        self.load_state_dict(th.load(path))

    #region training
    def learn(self, critic_only=False):
        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        value_epochs = []
        returns_epochs = []

        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            self.trajectory_sampler = self.trajectory_buffer.sample_trajectories(self.trajectory_sample_size)
            
            # Sample data from sampled trajectories
            for rollout_data in self.trajectory_buffer.sample(self.trajectory_sampler, self.data_sample_size,\
                self.batch_size, use_best_value=True, threshold=0):

                value_epochs.append(rollout_data.old_values)
                returns_epochs.append(rollout_data.returns)

                # for rollout data in rollout data:
                # if discrete
                cont_actions = rollout_data.cont_actions
                disc_actions = rollout_data.disc_actions.long().flatten()
                
                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    #self.policy.reset_noise(self.batch_size)
                    pass

                # evaluate actions
                values, log_prob, entropy = self.evaluate_actions(rollout_data.observations, cont_actions, disc_actions, rollout_data.cursors)
                values = values.flatten()

                # Normalize advantages
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1
                if len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > self.clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                values_pred = values
                value_loss = F.mse_loss(values_pred, rollout_data.returns)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                    entropy_losses.append(entropy_loss.item())


                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                #self._n_updates += 1
                if not continue_training:
                    break

                self.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()
        

        print(f"entropy loss: {np.mean(entropy_losses)}, policy loss: {np.mean(pg_losses)} value loss: {np.mean(value_losses)}")
        return np.mean(value_losses)

def tensor_stats(tensor, name="t"):
    t = tensor.detach()
    return f"{name} mean={t.mean().item():.6g} std={t.std().item():.6g} min={t.min().item():.6g} max={t.max().item():.6g}"

#region Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=0.99)
    args = parser.parse_args()

    device = "cuda" if th.cuda.is_available() else "cpu"
    buffer_size = 5
    trajectory_sample_size = 2

    obs_space = (4, 84, 84)
    cont_action_space = 2
    disc_action_space = 2
    host = "127.0.0.1"
    port = 13000
    n_episodes = 4000

    osuai = HP3O(device, buffer_size, trajectory_sample_size, obs_space, disc_action_space, cont_action_space)
    #osuai.load_model("tmp/hp3o_offline_pretrained.pt")
    osuenv = OsuEnv((host, port))

    osuenv.wait_for_game_to_launch()
    time.sleep(1)

    for episode in range(n_episodes):
        obs, cur_pos = osuenv.reset()
        #osuai.eval()

        score = 0
        t = 0
        done = False
        pbar = tqdm(desc=f"trajectory {episode+1} ", unit="frame", leave=False)
        while not done:
            obs = th.as_tensor(obs).to(device)
            cur_pos = th.tensor(cur_pos, dtype=th.float32, device=device)
            with th.no_grad():
                cont_action, disc_action, log_prob, value = osuai.choose_action(obs.unsqueeze(0), cur_pos.unsqueeze(0))
                next_obs, reward, done, next_cur_pos = osuenv.step(cont_action.cpu().numpy(), disc_action.cpu().item())
            
            #print(disc_action)
            
            # if reward != 0:
            #     print(f"reward: {reward} at timestep {t}")
            
            score+=reward
            osuai.add(obs, cont_action, disc_action, reward, done, value, log_prob, cur_pos)
            obs = next_obs
            cur_pos = next_cur_pos
            t+=1
            pbar.update(1)
        
        pbar.close()

        if len(osuai.trajectory_buffer.trajectories) >= trajectory_sample_size: 
            osuai.learn()
        
        if (episode+1) % 10 == 0:
            print("model saved")
            osuai.save_model()

        print(f"episode {episode+1} with length {t} score={score:.2f}") # cont_mean={cont_action.mean(axis=0)} cont_std={cont_action.std(axis=0)}