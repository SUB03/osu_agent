import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Independent
import gymnasium as gym
from matplotlib import pyplot as plt
import time

from Trajectory_Replay_Buffer import TrajectoryBuffer

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class HP3O(nn.Module):
    def __init__(self, device, buffer_size, trajectory_sample_size, obs_space, action_space):
        super(HP3O, self).__init__()
        self.buffer_size = buffer_size
        self.device = device
        self.n_epochs = 10
        self.gamma = 0.99
        self.gae_lambda = 0.92
        self.clip_range = 0.2 # original 0.1
        self.clip_range_vf = 0.2
        self.data_sample_size = 1280
        self.batch_size = 128
        self.target_kl = None
        self.ent_coef = 0.0004
        self.vf_coef = 0.5 #0.58096
        self.max_grad_norm = 0.8
        self.verbose = 1
        self.lr = 5e-4
        self.use_sde = False
        self.trajectory_sample_size = trajectory_sample_size
        self.trajectory_buffer = TrajectoryBuffer(buffer_size, obs_space, action_space, device, self.gamma, self.gae_lambda)

        self.actor = nn.Sequential(
            layer_init(nn.Linear(8, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, action_space), std=0.01),
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(8, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.),
        )

        self.logstd = nn.Parameter(torch.zeros(1, action_space))

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.to(device)

    def get_value(self, x):
        return self.critic(x)
    
    def choose_action(self, x):
        mu = self.actor(x)
        logstd = self.logstd.expand_as(mu)
        std = torch.exp(logstd) + 1e-8
        probs = Normal(mu, std)
        probs = Independent(probs, 1)
        action = probs.rsample()
        value = self.critic(x)
        return action.detach().cpu().squeeze(0).numpy(), probs.log_prob(action).detach(), value.detach().cpu()
    
    def evaluate_actions(self, obs, actions):
        mu = self.actor(obs)
        logstd = self.logstd.expand_as(mu)
        std = torch.exp(logstd)
        probs = Normal(mu, std)
        probs = Independent(probs, 1)
        return self.critic(obs), probs.log_prob(actions), probs.entropy()
    
    def add(self, obs, action, reward, done, value, log_prob):
        self.trajectory_buffer.add(obs, action, reward, done, value, log_prob)
    
    def learn(self):
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
                actions = rollout_data.actions#.long().flatten()
                
                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    #self.policy.reset_noise(self.batch_size)
                    pass

                # evaluate actions
                values, log_prob, entropy = self.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()

                # Normalize advantages
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1
                if len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                values_pred = values
                # NOTE: this depends on the reward scaling
                values_pred = rollout_data.old_values + torch.clamp(
                    values - rollout_data.old_values, -self.clip_range_vf, self.clip_range_vf
                )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                    entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # optimization step
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_prob) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                #self._n_updates += 1
                if not continue_training:
                    break

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()
            
        print(f"entropy loss: {np.mean(entropy_losses)}, policy loss: {np.mean(pg_losses)} value loss: {np.mean(value_losses)}")



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_episodes = 500
    buffer_size = 20
    trajectory_sample_size = 5
    obs_space = 8
    action_space = 2


    agent = HP3O(device, buffer_size, trajectory_sample_size, obs_space, action_space)
    env = gym.make("LunarLander-v3", continuous = True)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    scores = []

    for episode in range(num_episodes):
        obs, info, = env.reset()

        done = False
        while not done:
            obs = torch.tensor(obs).to(device)
            with torch.no_grad():
                action, log_prob, value = agent.choose_action(obs.unsqueeze(0))
                new_obs, reward, terminated, truncated, info = env.step(action)

            done = True if terminated or truncated else False
            agent.add(obs.cpu(), action, reward, done, value, log_prob)
            obs = new_obs


        if "episode" in info:
            reward = info['episode']['r']
            scores.append(reward)
            print(f"episode {episode} reward: {reward}")

        
        if len(agent.trajectory_buffer.trajectories) >= trajectory_sample_size:
            agent.learn()

    def moving_average(data, window_size):
        """Calculate the moving average of data using a specified window size."""
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    window_size = 10  # Choose a window size that works for your data
    smoothed_scores = moving_average(scores, window_size)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(scores, label='Original Scores', alpha=0.5)
    plt.plot(np.arange(window_size - 1, len(scores)), smoothed_scores, label='Smoothed Scores', color='red')
    plt.title('Rewards Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.savefig('rewards_hp3o+_final_test.png')
    plt.show()