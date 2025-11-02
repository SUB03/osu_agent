import os
import argparse
import numpy as np
from tqdm import tqdm
import torch as th
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Normal, Independent, Categorical
import torch.nn as nn
import torch.optim as optim

from HP3O_cont_osu import HP3O
from ReplayDataset import ReplayDataset


def train(args):
    device = 'cuda' if th.cuda.is_available() else 'cpu'
    
    model = HP3O(device, 20, 1280, (4, 60, 80), 2, 2)
    dataset = ReplayDataset(pkl=os.path.join(args.data_dir, "trajectories_index.pkl"))
    print(f"dataset samples: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    mse_loss_fn = nn.MSELoss()
    ce_loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for i, (obs, prev_cursors, tar_cursors, disc_targets) in enumerate(loader):
            obs = obs.to(device)
            prev_cursors = prev_cursors.to(device)
            tar_cursors = tar_cursors.to(device)
            disc_targets = disc_targets.to(device)

            # get predictions
            features = model.preprocess(obs)
            features = model.feature_extractor(features)
            features = th.cat((features, prev_cursors), dim=1)
            features = model.fc(features)

            disc_logits = model.disc_actor(features)
            mu = model.cont_actor(features)
            logstd = model.logstd(features)
            std = F.softplus(logstd)
            cont_dist = Independent(Normal(mu, std), 1)

            cont_pred = cont_dist.rsample()

            # calculate losses
            #cont_loss = mse_loss_fn(mu, tar_cursors)
            cont_loss = mse_loss_fn(cont_pred, tar_cursors)
            disc_loss = ce_loss_fn(disc_logits, disc_targets)

            # optimize
            #print(value_loss)
            optimizer.zero_grad()
            (cont_loss + 0.2 * disc_loss).backward()
            th.nn.utils.clip_grad_norm_(model.parameters(), model.max_grad_norm)
            optimizer.step()

            epoch_loss += cont_loss.item()
        
        print(f"Epoch: {epoch+1}/{args.epochs}, avg_loss: {np.mean(epoch_loss):6f}")   

    # save model after training
    th.save(model.state_dict(), args.save_model)
    print(f"Model saved to {args.save_model}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=".data", help="folder where trajectories are stored")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2.5e-4, help="learning rate")
    parser.add_argument("--save_model", type=str, default="tmp/hp3o_offline_pretrained.pt")
    args = parser.parse_args()

    train(args)