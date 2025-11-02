import argparse, os, time
from datetime import datetime
import numpy as np
import pandas as pd

from OsuEnv import OsuEnv

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def make_dataset(args):
    ensure_dir(args.out_dir)
    traj_dir = os.path.join(args.out_dir, "trajectories")
    ensure_dir(traj_dir)

    env = OsuEnv((args.host, args.port))
    print("waiting for game to start")
    env.wait_for_game_to_launch()

    current_date = datetime.now().strftime("%Y_%m_%d")
    rows = []

    for i in range(args.n_trajs):
        traj = env.collect_trajectory(i)

        fname = os.path.join(traj_dir, f"traj_{i:04d}_{current_date}.npz")
        np.savez(fname,
            observations=traj["observations"],
            prev_mouse_pos=traj["prev_mouse_pos"],
            tar_mouse_pos=traj["tar_mouse_pos"],
            disc_actions=traj["discrete_actions"],
            rewards=traj["rewards"])

        
        rows.append({
            "traj_id": i,
            "file": fname,
        })

        time.sleep(4)

    df = pd.DataFrame(rows)
    df_path = os.path.join(args.out_dir, "trajectories_index.pkl")
    df.to_pickle(df_path)
    print("Done collecting")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=".data", help="output directory where collected trajectoris will be stored")
    parser.add_argument("--n_trajs", type=int, default=3, help="number of trajectories to collect")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="host ip to your data reader")
    parser.add_argument("--port", type=int, default=13000, help="port for your data reader")
    args = parser.parse_args()

    make_dataset(args)