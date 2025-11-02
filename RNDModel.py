# put near the top with imports
import torch as th
import torch.nn as nn
import math

class RNDModel(nn.Module):
    def __init__(self, input_shape, output_size=128):
        super().__init__()
        c, h, w = input_shape
        self.target = nn.Sequential(
            nn.Conv2d(c, 32, 3, stride=2), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2), nn.SiLU(),
            nn.Flatten(),
            nn.Linear(17024, output_size)
        )
        self.predictor = nn.Sequential(
            nn.Conv2d(c, 32, 3, stride=2), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2), nn.SiLU(),
            nn.Flatten(),
            nn.Linear(17024, 512), nn.SiLU(),
            nn.Linear(512, output_size)
        )
        # freeze target
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, obs):
        # obs: (B, C, H, W)
        t = self.target(obs)
        p = self.predictor(obs)
        return p, t
