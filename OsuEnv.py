import pydirectinput
import pygetwindow as gw
import utils, socket, json, time, threading
from collections import deque
from tqdm import tqdm
import numpy as np
import torch
from pynput import mouse
import cv2
import mss

class OsuEnv:
    def __init__(self, localhost, n_frame_to_stack = 4, target_agent_fps = 30):
        self.n_frame_to_stack = n_frame_to_stack
        self.screen_width, self.screen_height = pydirectinput.size()
        self.left, self.top, self.right, self.bottom = 300, 100, 1600, 1000
        self.width = self.right - self.left
        self.height = self.bottom - self.top
        self.delay = 1/30
        self.max_distance = 400
        self.monitor = {"left": self.left, "top": self.top, "width": self.width, "height": self.height}
        self.frame_buffer = deque(maxlen=n_frame_to_stack)

        self.tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_server.connect(localhost)
        self.tcp_server.settimeout(0.5)
        pydirectinput.FAILSAFE = False

        self._click_lock = threading.Lock()
        self.button_state = 0
        self.timestep = 0
        self.data = self.get_snapshot()
    
    def capture_cursor_pos(self):
        x, y = pydirectinput.position()
        # normalize to [0,1] relative to your monitor region
        nx = (x - self.left) / self.width
        ny = (y - self.top) / self.height
        return np.array([nx, ny], dtype=np.float32)
    
    def capture_frame(self):
        with mss.mss() as sct:
            img = sct.grab(self.monitor)

        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (84, 84))
        img = np.array(img).astype(np.float32) / 255.0

        return img

    def step(self, continuous_action, discrete_action):
        start_time = time.monotonic()
        if not utils.is_osu_active_window():
            return


        self.scale_mouse_action(continuous_action)
        #self.scale_mouse_action_relative(continuous_action)
        action = int(discrete_action)

        if action == 1:
            pydirectinput.mouseDown(_pause = False)
        else:
            pydirectinput.mouseUp(_pause = False)


        new_data = self.get_snapshot()

        reward = self.calculate_reward(new_data['hits'], self.data['hits'], new_data['combo'], self.data['combo'])
          
        # if new_data['hp'] == 0 and self.timestep > 110:
        #     #print(f'died at {self.timestep}')
        #     done = True
        #     reward = 0
        if new_data['status'] == 7:
            done = True
            reward = 0
        else:
            done = False

        self.timestep+=1
        self.data = new_data

        img = self.capture_frame()
        self.frame_buffer.append(img)
        frames = np.stack(list(self.frame_buffer), axis=0)

        cursor_pos = self.capture_cursor_pos()

        self.next_frame_time += self.delay
        sleep_time = self.next_frame_time - time.monotonic()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            self.next_frame_time = time.monotonic()

        return frames, reward, done, cursor_pos
    
    def reset(self):
        self.frame_buffer.clear()
        curr_data = self.get_snapshot()

           # Activate osu! window
        osu = gw.getWindowsWithTitle("osu!")[0]
        if osu.isMinimized:
            osu.restore()

        osu.activate()


        if curr_data['status'] == 7: # result screen
            pydirectinput.moveTo(self.screen_width//2, self.screen_height//2)
            time.sleep(4)
            pydirectinput.press('esc')
            time.sleep(1)
        elif curr_data['hp'] == 0 and curr_data['status'] == 2:
            pydirectinput.moveTo(self.screen_width//2, self.screen_height//2)
            time.sleep(4)
            pydirectinput.press("esc")
        # elif self.curr_data['status'] == 0: # 0 menu 5 song select
        #     pydirectinput.moveTo(self.screen_width//2, self.screen_height//2)
        #     time.sleep(2)
        #     pydirectinput.press("esc")

        utils.make_random_choice()
        self.wait_for_map_to_start()

        for _ in range(self.n_frame_to_stack):
            img = self.capture_frame()
            self.frame_buffer.append(img)

        self.timestep = 0
        self.data = self.get_snapshot()
        frames = np.stack(list(self.frame_buffer), axis=0)
        cursor_pos = self.capture_cursor_pos()
        self.next_frame_time = time.monotonic()
        return frames, cursor_pos

    def scale_mouse_action(self, continuous_action):
        # Normalize to [-1, 1] -> [0, width/height] -> offset into playfield
        #continuous_action = np.tanh(continuous_action)
        continuous_action = np.clip(continuous_action, -1, 1)
        x = int((continuous_action[0] + 1) * self.width / 2) + self.left
        y = int((continuous_action[1] + 1) * self.height / 2) + self.top

        pydirectinput.moveTo(x, y, _pause = False)

    def scale_mouse_action_relative(self, continuous_action):
        # ensure array and clip
        cont = np.asarray(continuous_action, dtype=np.float32)
        if cont.size < 2:
            return
        cont = np.clip(cont[:2], -1.0, 1.0)

        # map to pixel deltas
        dx = int(cont[0] * self.max_distance)
        dy = int(cont[1] * self.max_distance)

        if dx == 0 and dy == 0:
            return

        # current cursor position
        cur_x, cur_y = pydirectinput.position()

        # compute new absolute target
        min_x, max_x = self.left, self.left + self.width - 1
        min_y, max_y = self.top, self.top + self.height - 1
        target_x = int(np.clip(cur_x + dx, min_x, max_x))
        target_y = int(np.clip(cur_y + dy, min_y, max_y))

        # warp cursor instantly
        pydirectinput.moveTo(target_x, target_y, _pause=False)

    
    def calculate_reward(self, new_hits, old_hits, new_combo, combo):
        weights = {
            'hit300': 10,
            'hit100': 5,
            'hit50': 1,
            'misses': 0
        }

        reward = 0
        for key in weights:
            reward += (new_hits.get(key, 0) - old_hits.get(key, 0)) * weights[key]
        
        self.hits = new_hits
        if reward == 0 and combo > new_combo:
            reward -= 1

        if reward == 0 and combo < new_combo:
            reward += 1

        # combo_scale = 0.5 
        # if combo <= 1:
        #     combo_factor = 1.0
        # else:
        #     # using log1p ensures smooth, sublinear growth (gentle at large combo)
        #     combo_factor = 1.0 + combo_scale * np.log1p(combo - 1)
        #print(f"combo factor: {combo_factor}")
        #print(f"reward: {reward}")
        # reward_bonus = 0
        # if combo >= 1:
        #     reward_bonus = 0.01


        return reward #* combo_factor
    
    def get_snapshot(self):
        self.tcp_server.sendall(b"GetSnapShot")
        msg = self.tcp_server.recv(1024)
        msg = json.loads(msg)
        return msg
    
    
    def wait_for_game_to_launch(self, delay = 0.05):
        while True:
            try:
                msg = self.get_snapshot()
                
                # 2 if playing 0 is in menu, -1 is not in game
                if msg['status'] != -1:
                    break
            except Exception as e:
                print(e)
            
            time.sleep(delay)
    
    def wait_for_map_to_start(self, delay = 0.005):
        while True:
            msg = self.get_snapshot()
            # 2 if mplaing 5 is in menu, -1 is not in game
            if msg['status'] == 2 and msg['displayedPlayerHp'] > 150:
                return 

            time.sleep(delay)
    
    def close(self):
        self.tcp_server.close()

    def on_click(self, x, y, button, pressed):
        try:
            if button == mouse.Button.left:
                #ts = int(time.time() * 1000)
                with self._click_lock:
                    self.button_state = 1 if pressed else 0
        except Exception:
            pass

    def collect_trajectory(self, traj_id):
        listener = mouse.Listener(on_click=self.on_click)
        listener.daemon = True

        observations = []
        prev_mouse_pos = []
        tar_mouse_pos = []
        discrete_actions = []
        rewards = []

        print("waiting for map to start")   

        obs, mouse_pos = self.reset()
        pbar = tqdm(desc=f"trajectory {traj_id}", unit="frame", leave=False)
        with self._click_lock:
            disc_action = self.button_state
        listener.start()

        try:
            while True:
                start_time = time.monotonic()
                new_snapshot = self.get_snapshot()
                new_mouse_pos = self.capture_cursor_pos()
                status = new_snapshot["status"]
                obs = np.stack(list(self.frame_buffer), axis=0)
                mouse_pos = self.capture_cursor_pos()
                #print(disc_action)

                prev_mouse_pos.append(mouse_pos)
                tar_mouse_pos.append(new_mouse_pos)
                discrete_actions.append(disc_action)
                observations.append(obs)
                rewards.append(self.calculate_reward(new_snapshot["hits"], self.data["hits"], new_snapshot["combo"], self.data["combo"]))

                pbar.update(1)
                self.data = new_snapshot
                mouse_pos = new_mouse_pos

                if status != 2:
                    pbar.close()
                    break

                self.frame_buffer.append(self.capture_frame())
                with self._click_lock:
                    disc_action = self.button_state
                
                # frame limiter
                self.next_frame_time += self.delay
                sleep_time = self.next_frame_time - start_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    self.next_frame_time = time.monotonic()
            
        finally:
            try:
                listener.stop()
            except Exception:
                pass

        
        observations = np.stack(observations, axis=0)
        prev_mouse_pos = np.stack(prev_mouse_pos, axis=0)
        tar_mouse_pos = np.stack(tar_mouse_pos, axis=0)
        discrete_actions = np.stack(discrete_actions, axis=0)
        rewards = np.array(rewards, dtype=np.float32)

        return {
            "observations": observations,
            "prev_mouse_pos": prev_mouse_pos,
            "tar_mouse_pos": tar_mouse_pos,
            "discrete_actions": discrete_actions,
            "rewards": rewards
        }


if __name__ == "__main__":
    osuenv = OsuEnv(("127.0.0.1", 13000))
    print(osuenv.calculate_reward({
            'hit300': 1,
            'hit100': 0,
            'hit50': 0,
            'misses': 1
        }, osuenv.curr_data['hits'], 400))
    
    print(osuenv.calculate_reward(osuenv.curr_data['hits'], osuenv.curr_data['hits'], 1))
    
    print(f"combo: {osuenv.curr_data['combo']}")
    print(f"hits: {osuenv.curr_data['hits']}")