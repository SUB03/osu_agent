import torch as th
import numpy as np
from HP3O_Hybrid_action import HP3O
from OsuEnv import OsuEnv
import utils
from matplotlib import pyplot as plt
import pygetwindow as gw
import time

host = "127.0.0.1"
port = 13000

device = 'cuda' if th.cuda.is_available() else 'cpu'
trajectory_buffer_size = 10
trajectory_sample_size = 5
frames_to_stack = 4
obs_space = (frames_to_stack, 84, 84)
disc_action_space = 3
cont_action_space = 2

n_episodes = 5000

osuai = HP3O(
    device,
    trajectory_buffer_size,
    trajectory_sample_size,
    obs_space,
    disc_action_space,
    cont_action_space
)
#osuai.load_model()

osuenv = OsuEnv((host, port), frames_to_stack)

osuenv.wait_for_game_to_launch()
# Activate osu! window
osu = gw.getWindowsWithTitle("osu!")[0]
if osu.isMinimized:
    osu.restore()

osu.activate()
time.sleep(1)

rewards = []
for episode in range(n_episodes):
    obs = osuenv.reset()
    #osuai.eval()

    score = 0
    t = 0
    done = False
    while not done:
        with th.no_grad():
            disc_action, cont_action, log_prob, value = osuai.choose_action(th.as_tensor(obs).to(device).unsqueeze(0))
            next_obs, reward, done = osuenv.step(cont_action, disc_action)
        
        #print(f"reward: {reward} at timestep {t}, with action {disc_action}")
        

        score+=reward
        osuai.add(obs, disc_action, cont_action, reward, done, value, log_prob)
        obs = next_obs     
        t+=1
    if len(osuai.trajectory_buffer.trajectories) >= trajectory_sample_size:       
        osuai.learn()

        if episode % 25 == 0:

            print("model saved")    
            osuai.save_model()

    print(f"episode {episode} score={score:.2f}") # cont_mean={cont_action.mean(axis=0)} cont_std={cont_action.std(axis=0)}
osuai.save_model()