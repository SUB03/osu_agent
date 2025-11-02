import os
import numpy as np
import torch as torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, input_size, n_actions, device, lr=5e-4):
        super(Agent, self).__init__()
        self.lr = lr
        self.gamma = 0.99
        self.device = device

        self.discrete_actor = nn.Sequential(
            layer_init(nn.Linear(input_size, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, n_actions), std=0.01),
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(input_size, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.),
        )

        self.actor_optimizer = optim.Adam(self.discrete_actor.parameters(), lr=lr, betas=(0.999, 0.999))
        self.critic_optimizer = optim.Adagrad(self.critic.parameters(), lr=lr, betas=(0.999, 0.999))

    def get_value(self, x):
        return self.critic(x)
    
    def choose_action(self, x, action = None):
        logits = self.discrete_actor(x)
        discrete_probs = Categorical(logits=logits)
        if action == None:
            action = discrete_probs.sample()
        return action.squeeze(0), discrete_probs.log_prob(action), discrete_probs.entropy()
    
    def learn(self, state_batch, actions_batch, log_probs_batch, reward_batch, next_state_batch, dones_batch, batch_size):

        # calc returns - Monte Carlo and advantages
        returns = torch.zeros((batch_size))
        advantages = []
        T = len(reward_batch)
        G = torch.zeros(T, device=self.device)
        G[-1] = reward_batch[-1]
        for t in reversed(range(T-1)):
            G[t] = reward_batch[t] + self.gamma * G[t+1]
        returns.append(G)
        with torch.no_grad():
            values = self.get_value(state_batch).squeeze()
        advantages.append(G - values)

        _, newlogprob, entropy, newvalue = agent.choose_action(state_batch, actions_batch)
        ratio = (newlogprob - log_probs_batch).exp()

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
        

if __name__ == "__main__":
    agent = Agent(n_epochs=1)
    print("agent loaded")
