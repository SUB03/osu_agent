import re
import os
import time
import difflib
import socket
import pygetwindow as gw
import pyautogui
import pydirectinput

pattern = re.compile(r" \[[a-zA-Z'\s.!@#$%^&*(),?\":{}|<>]+\]")

def get_parsing(maps: list, osu_songs_directory: str, title: str):
    suffix = re.findall(pattern, title)[-1]
    # for m in maps:
    #     if title[:-len(suffix)] in m:
    #         map_path = os.path.join(osu_songs_directory, m)

    """We use slower algorithm to match map names with some similarity only
                because of some map's inaccurate names"""
    match = difflib.get_close_matches(title[:-len(suffix)], maps, 1, 0.8)[0]
    map_path = os.path.join(osu_songs_directory, match)

    #print(map_path)
    if map_path == None:
        print("Map not found for some reason")

    file = [x for x in os.listdir(map_path) if x.endswith(f"{suffix}.osu")][0]
    #file = [x for x in os.listdir(map_path) if x.endswith(".osu")][0]

    osu_path = os.path.join(map_path, file)
    
    return osu_path


def make_random_choice():
    if gw.getActiveWindow().title != "osu!":
        print("im the reason")
        return
    
    #print(gw.getActiveWindow().title)
    pydirectinput.press('f2')
    #print("f2")
    time.sleep(0.5)
    pydirectinput.press('enter')
    #print("enter")
    #time.sleep(0.5)

    
def is_osu_active_window():
    return gw.getActiveWindowTitle()[:4] == "osu!"

# if __name__ == "__main__":
#     wait_for_game_to_launch(("127.0.0.1", 13000))