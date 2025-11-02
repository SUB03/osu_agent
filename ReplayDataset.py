import os
import pandas as pd
import torch as th
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ReplayDataset(Dataset):
    def __init__(self, pkl):
        self.df = pd.read_pickle(pkl)
        self.observations_samples = []
        self.prev_mouse_samples = []
        self.tar_mouse_samples = []
        self.disc_actions = []

        # Walk each trajectory and build sample references
        for _, row in self.df.iterrows():
            traj_file = str(row["file"])
            if not os.path.exists(traj_file):
                print("Warning: trajectory file not found:", traj_file)
                continue
            data = np.load(traj_file)
            self.observations_samples.append(data['observations'])
            self.prev_mouse_samples.append(data['prev_mouse_pos'])
            self.tar_mouse_samples.append(data['tar_mouse_pos'])
            self.disc_actions.append(data['disc_actions'])
            data.close()

        self.observations_samples = th.tensor(np.concatenate(self.observations_samples, axis=0), dtype=th.float32)
        self.prev_mouse_samples = th.tensor(np.concatenate(self.prev_mouse_samples, axis=0), dtype=th.float32)
        self.tar_mouse_samples = th.tensor(np.concatenate(self.tar_mouse_samples, axis=0), dtype=th.float32)
        self.disc_actions_samples = th.tensor(np.concatenate(self.disc_actions, axis=0), dtype=th.long)

        # print(np.bincount(self.disc_actions_samples))
        # raise

    def __len__(self):
        return len(self.observations_samples)

    def __getitem__(self, idx):
        # convert cursors -> [-1,1] as model expects
        prev_cursor = (self.prev_mouse_samples[idx] * 2.0) - 1.0
        target_cursor = (self.tar_mouse_samples[idx] * 2.0) - 1.0

        return self.observations_samples[idx], prev_cursor, target_cursor, self.disc_actions_samples[idx]