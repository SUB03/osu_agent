# from mss import mss
# from PIL import Image
# import cv2
# import time
# import numpy as np

# class FrameCapture():
#     def __init__(self):
#         self.sct = mss()
#         self.monitor = self.sct.monitors[1]
#         self.frame_shape = (self.monitor['height'], self.monitor['width'])

#     def take_screenshot(self):
#         img = self.sct.grab(self.monitor)
#         img = np.array(img)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         # img = Image.frombytes('RGB', self.frame_shape, img.rgb)
#         # img = img.convert('L')
#         return img / 255.0

#     def close(self):
#         self.sct.close()

# if __name__ == '__main__':
#     frame_capture = FrameCapture()
#     frames = frame_capture.stack_frames()
#     frame_capture.close()

#     print(frames.shape)