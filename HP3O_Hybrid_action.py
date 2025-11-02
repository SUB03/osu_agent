import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, Independent
import torchvision.transforms.v2 as v2
from matplotlib import pyplot as plt

from Trajectory_Replay_Buffer_hybrid_action_space import TrajectoryBuffer

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class HP3O(nn.Module):
    def __init__(self, device, buffer_size, trajectory_sample_size, obs_space, disc_action_space,\
            cont_action_space):
        super(HP3O, self).__init__()

        self.buffer_size = buffer_size
        self.trajectory_sample_size = trajectory_sample_size
        self.device = device
        self.n_epochs = 10
        self.gamma = 0.99
        self.gae_lambda = 0.92
        self.clip_range_vf = None
        self.clip_range = 0.1 # original 0.1
        self.data_sample_size = 1280
        self.batch_size = 128
        self.target_kl = None
        self.ent_coef = 0.0004 #0.0004
        self.vf_coef = 0.58096 #0.58096
        self.max_grad_norm = 0.8
        self.verbose = 1
        self.lr = 1e-4
        
        self.runn_mean = RunningMeanStd()
        self.trajectory_buffer = TrajectoryBuffer(buffer_size, obs_space, 1, cont_action_space, device, self.gamma, self.gae_lambda)

        self.preprocess = nn.Sequential(
            layer_init(nn.Conv2d(4, 64, 5, 3)),
            #nn.BatchNorm2d(64),
            nn.ReLU(),

            layer_init(nn.Conv2d(64, 128, 5, 2)),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            
            layer_init(nn.Conv2d(128, 128, 3, 1)),
            #nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros((1, *obs_space))
            feature_size = self.preprocess(dummy).size(1)

        self.shared = nn.Sequential(
            layer_init(nn.Linear(feature_size, 2048)),
            nn.ReLU(),
            layer_init(nn.Linear(2048, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 256)),
            nn.ReLU(),
        )


        self.discrete_actor = nn.Sequential(
            layer_init(nn.Linear(256, disc_action_space), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(256, 1), std=1.)
        )
        # self.continuous_features = nn.Sequential(
        #     layer_init(nn.Linear(feature_size, 512)),
        #     nn.SiLU(),
        #     layer_init(nn.Linear(512, 256)),
        #     nn.SiLU(),
        # )
        self.mu = layer_init(nn.Linear(256, cont_action_space), std=0.01)
        self.logstd = layer_init(nn.Parameter(1, cont_action_space), std=0.01)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.to(device)

    def get_value(self, obs):
        features = self.preprocess(obs)
        return self.critic(features)
    
    def get_dist(self, mu, logstd):
        std = torch.exp(logstd) + 1e-6  # Small noise for stability
        #logstd = torch.clamp(logstd, min=-20.0, max=5)
        base_dist = Normal(mu, std)
        base_dist = Independent(base_dist, 1)
        return base_dist

    
    def choose_action(self, obs):
        features = self.preprocess(obs)
        features = self.shared(features)
        #cont_features = self.continuous_features(features)

        disc_logits = self.discrete_actor(features)
        value = self.critic(features)

        disc_probs = Categorical(logits=disc_logits)
        cont_probs = self.get_dist(self.mu(features), self.logstd.expand_as(features))

        disc_action = disc_probs.sample()
        cont_action = cont_probs.rsample()

        total_prob = disc_probs.log_prob(disc_action) +\
            cont_probs.log_prob(cont_action)
        
        return disc_action.detach().cpu().numpy(), cont_action.detach().squeeze(0).cpu().numpy(), total_prob.detach().item(), value.detach().squeeze().item()
    
    def evaluate_actions(self, obs, disc_actions, cont_actions):
        features = self.preprocess(obs)
        features = self.shared(features)
        #cont_features = self.continuous_features(features)
        disc_logits = self.discrete_actor(features)
        
        disc_probs = Categorical(logits=disc_logits)
        cont_probs = self.get_dist(self.mu(features), self.logstd(features))

        total_log_prob = disc_probs.log_prob(disc_actions) +\
           cont_probs.log_prob(cont_actions)
        
        disc_ent = disc_probs.entropy()
        cont_ent = cont_probs.entropy()
        total_entropy = disc_ent +\
           cont_ent
        
        # diagnostic
        with torch.no_grad():
            disc_ent = disc_ent.detach().cpu().mean().item()
            cont_ent = cont_ent.detach().cpu().mean().item()
        
        return self.critic(features), total_log_prob, total_entropy,\
            disc_ent, cont_ent,
    
    def add(self, obs, disc_action, cont_action, reward, done, value, log_prob):
        self.trajectory_buffer.add(obs, disc_action, cont_action, reward, done, value, log_prob)

    def save_model(self):
        torch.save(self.state_dict(), "tmp/HP3O_osu_v1.pt")

    def load_model(self):
        self.load_state_dict(torch.load("tmp/HP3O_osu_v1.pt"))
    
    def learn(self):
        # effects layers like dropout, batchnorm
        self.train()

        entropy_losses = []
        cont_entropies = []
        disc_entropies = []
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
                
                returns = rollout_data.returns
                returns_np = returns.cpu().numpy()
                self.runn_mean.update(returns_np)
                returns = torch.tensor(self.runn_mean.normalize(returns_np), device=returns.device, dtype=returns.dtype)
                disc_actions = rollout_data.disc_actions.long().flatten()
                cont_actions = rollout_data.cont_actions
                
                # # Re-sample the noise matrix because the log_std has changed
                # TODO: implement
                # if self.use_sde:
                #     self.reset_noise(self.batch_size)

                # evaluate actions
                values, log_prob, entropy, disc_ent, cont_ent = \
                    self.evaluate_actions(rollout_data.observations, disc_actions, cont_actions)
                values = values.flatten()

                cont_entropies.append(cont_ent)
                disc_entropies.append(disc_ent)

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
                
                if self.clip_range_vf is None:
                    values_pred = values
                # NOTE: this depends on the reward scaling
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -self.clip_range_vf, self.clip_range_vf
                    )
                value_loss = F.mse_loss(returns, values_pred)
                value_losses.append(value_loss.item())

                
                entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # optimization step
                # with torch.no_grad():
                #     log_ratio = log_prob - rollout_data.old_log_prob
                #     approx_kl_div = torch.mean((torch.exp(log_prob) - 1) - log_ratio).cpu().numpy()
                #     approx_kl_divs.append(approx_kl_div)

                # if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                #     continue_training = False
                #     if self.verbose >= 1:
                #         print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                #     break

                #self._n_updates += 1
                # if not continue_training:
                #     break

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()

                disc_actor_norm = torch.norm(torch.cat([p.grad.flatten() for p in self.discrete_actor.parameters() if p.grad is not None]))
                #cont_feat_norm = torch.norm(torch.cat([p.grad.flatten() for p in self.continuous_features.parameters() if p.grad is not None]))
                cont_mean_norm = torch.norm(torch.cat([p.grad.flatten() for p in self.mu.parameters() if p.grad is not None]))
                cont_std_norm = torch.norm(torch.cat([p.grad.flatten() for p in self.logstd.parameters() if p.grad is not None]))
                critic_norm = torch.norm(torch.cat([p.grad.flatten() for p in self.critic.parameters() if p.grad is not None]))
                print("===DBG BATCH===")
                print("returns mean/std:", float(returns.mean()), float(returns.std()))
                print("values mean/std:", float(values.mean()), float(values.std()))
                print("old_values mean/std:", float(rollout_data.old_values.mean()), float(rollout_data.old_values.std()))
                print("advantages mean/std:", float(advantages.mean()), float(advantages.std()))
                print("log_prob mean/std:", float(log_prob.mean()), float(log_prob.std()))
                #print("old_log_prob mean/std:", float(rollout_data.old_log_prob.mean()), float(rollout_data.old_log_prob.std()))
                #print("ratio mean/std:", float(ratio.mean()), float(ratio.std()))
                #print("entropy mean:", float(entropy.mean()))
                #print(f"Disc actor Grad Norm: {disc_actor_norm:.4f}, Critic Grad Norm: {critic_norm:.4f}")
                #print(f"Cont feat Grad Norm: {cont_feat_norm:.4f}")
                #print(f"Cont mean Grad Norm: {cont_mean_norm:.4f}")
                #print(f"Cont logstd Grad Norm: {cont_std_norm:.4f}")
                print(f"Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}")

        #self.scheduler.step(np.mean(returns_epochs))
        print(f"mean entropy loss: {np.mean(entropy_losses):.6f}, discrete entropy: {np.mean(disc_entropies):.6f}, continuous entropy: {np.mean(cont_entropies):.6f}")

            
import numpy as np

class RunningMeanStd:
    def __init__(self, eps=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = eps

    def update(self, x):
        # x: numpy array, shape (N, *shape) or (N,)
        x = np.asarray(x, dtype='float64')
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        # Welford style batch update
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x, clip=10.0):
        x = np.asarray(x, dtype='float64')
        std = np.sqrt(self.var + 1e-8)
        return np.clip((x - self.mean) / std, -clip, clip)
    

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    replay_buffer_size = 20
    replay_buffer_sample_size = 5
    obs_space = (4, 84, 84)
    disc_action_space = 3
    cont_action_space = 2
    #torch.manual_seed(0) # for debugging

    osuai = HP3O(
        device,
        replay_buffer_size,
        replay_buffer_sample_size,
        obs_space,
        disc_action_space,
        cont_action_space,
        )
    
    #osuai.eval()
    with torch.no_grad():
        dummy = np.zeros((1, *obs_space), dtype=np.float32)
        disc_action, cont_action, log_prob, value = osuai.choose_action(torch.tensor(dummy, device=device))
    
    print(log_prob)
    raise
    
    osuai.add(dummy, disc_action, cont_action, 0, True, value, log_prob)
    disc_action, cont_action = torch.tensor(disc_action, device=device), torch.tensor(cont_action, device=device)

    osuai.evaluate_actions(torch.tensor(dummy, device=device), disc_action, cont_action)