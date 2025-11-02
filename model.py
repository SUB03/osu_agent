import torch
import torch.nn as nn
from torchvision.transforms import v2
import numpy as np
from mss import mss
from PIL import Image

import utils


class OsuModel(nn.Module):
    def __init__(self, device):
        super(OsuModel, self).__init__()

        self.device = device

        self.transforms = v2.Compose([
            v2.ToImage(),
            v2.Resize((225, 225)),
            v2.ToDtype(torch.float32, scale=True)
        ])

        self.sequential = nn.Sequential(
            nn.Conv2d(4, 64, 5, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, 2),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(6912, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
        )


        self.cord_fc = nn.Linear(128, 2)
        self.click_fc = nn.Linear(128, 1)

        self.to(self.device)
    
    def forward(self):
        X = self.transform_screenshot()
        X = self.sequential(X)
        cord_logits = torch.sigmoid(self.cord_fc(X))
        click_logits = torch.sigmoid(self.click_fc(X))
        return torch.cat(cord_logits, click_logits(X), dim=1)
    
    def train(self): 
        self.train()
        
        x = self.take_screenshot()
        x = self.transforms(x)

    @staticmethod
    def take_screenshot():
        if not utils.is_osu_active_window():
            return
        
        with mss() as sct:
            x = sct.grab(sct.monitors[1])
            x = Image.frombytes('RGB', (x.width, x.height), x.rgb)
            x = x.convert('L')
        
        return x
    
if __name__ == "__main__":
    scs = OsuModel.take_screenshot()
    