import os
import numpy as np
import datetime
from random import shuffle
import torch
import time

from python_osu_parser import *

import pygetwindow as gw

from OsuEnv import OsuEnv
from PPO import OsuAI
from frameCapture import FrameCapture

import utils
HOST = "127.0.0.1"
PORT = 13000

osu_songs_directory = "D:\Games\osu!\Songs"
maps = os.listdir(osu_songs_directory)

print("waiting for game to launch...")
utils.wait_for_game_to_launch((HOST, PORT))
osuEnv = OsuEnv((HOST, PORT))
osuAI = OsuAI()

#get map name from window title
prefix = "osu!"
osu = gw.getWindowsWithTitle(prefix)[0]
title = osu.title

if osu.isMinimized:
    osu.restore()
    
osu.activate()

time.sleep(1)

# choose random map
utils.make_random_choice()

print("waiting for map...")
while title == prefix:
    title = gw.getWindowsWithTitle(prefix)[0].title
    
if len(title) > len(prefix):
    title = title[8:]

print(title)

# osu_path = utils.get_parsing(maps, osu_songs_directory, title)
# parser = beatmapparser.BeatmapParser()

# Parse File
# time = datetime.datetime.now()
# parser.parseFile(osu_path)
# print("Parsing done. Time: ", (datetime.datetime.now() - time).microseconds / 1000, 'ms')

#Build Beatmap
# time = datetime.datetime.now()
# parser.build_beatmap()
# print("Building done. Time: ", (datetime.datetime.now() - time).microseconds / 1000, 'ms\n')

# arr = np.array(parser.beatmap["hitObjects"])

# print(arr.shape)
# for i in range(5):
#     print(arr[i]['startTime'])

print("Waiting for map to start...")
utils.wait_for_map_to_start((HOST, PORT))
for i in range(3):
    state = osuEnv.reset()
    while(True):
        state = osuAI.transformImage(torch.tensor(state)).to("cuda")
        action, _, _ = osuAI.choose_action(state)

        state, _, done = osuEnv.step(action)
        if done:
            break
    print(i)
osuEnv.close()
print("Program stoped")