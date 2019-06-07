import sys
sys.path.insert(0, "..")
# print(sys.path)

import tkinter as tk
import enroll,recogn,veri
from gaitnet import gaitnet
from mrcnn import process as mrcnn
import cv2
from utils.helper import *
from PIL import Image
import numpy as np
import torch
import argparse
parser = argparse.ArgumentParser()
# parser.add_argument('--device', type=str, default='cuda', help=" 'cuda' for GPU, 'cpu' for cpu")
parser.add_argument('--gpu_idx', type=int, default=0)
parser.add_argument('--camera_idx', type=int, default=0)
parser.add_argument('--input_height', type=int, default=960, help='the height of the input imag')
parser.add_argument('--input_width', type=int, default=540, help='the width of the input image')
parser.add_argument('--threshold', type=float, default=0.7, help='threshold for recognition')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_idx)

if torch.cuda.is_available():
    print("Device:", 'cuda')
    print("GPU Index:", args.gpu_idx)
else:
    print("Device:", 'cpu')

print("Input Video Resolution:", (args.input_width, args.input_height) )
print("Camera Index:", args.camera_idx )
class VideoCapture:
    def __init__(self, video_source):
        self.resolution = (args.input_width, args.input_height)

        self.vid = cv2.VideoCapture(video_source)
        self.vid.set(3, self.resolution[1])
        self.vid.set(4, self.resolution[0])
        # self.vid = cv2.VideoCapture('C:/Users/Tony/Dropbox/GaitNet/PRODUCT/GaitNet_DEV/testing_2019/Camera/01_01.mp4')
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source, please check your camera settings", video_source)
        # Get video source width and height
        # self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        # self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.width = self.resolution[1]
        self.height = self.resolution[0]
        self.cnt = 0

    def get_frame(self): # simulation from video files
        if self.vid.isOpened():
            self.cnt+=1
            ret, frame = self.vid.read()
            if self.cnt<=2:
                return (None, None)
            else:
                self.cnt=0
            if ret:
                frame = np.array(Image.fromarray(frame).resize((self.resolution[1], self.resolution[0]), Image.ANTIALIAS))
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return (None, None)

    # def get_frame(self):
    #     if self.vid.isOpened():
    #         ret, frame = self.vid.read()
    #         print(frame.shape)
    #         cv2.waitKey(100)
    #         if ret:
    #             return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    #     return (None, None)
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

vid = VideoCapture(args.camera_idx)
# vid = VideoCapture("C:/Users/Hao/Videos/WIN_20190517_14_36_42_Pro.mp4")

database = load_database()
def check_database():
    if len(database) == 0:
        print("Database is empty")
        print()
    print("Database:")
    for k, v in database.items():
        print(k)
def create_enroll_wind():
    enroll.main(database, vid, mrcnn, gaitnet, args)
def create_verifi_wind():
    veri.main(database, vid, mrcnn, gaitnet, args)
def create_recogn_wind():
    recogn.main(database, vid, mrcnn, gaitnet, args)
root = tk.Tk()
root.title("GaitNet Demo")
check_btn = tk.Button(root, text="Check Database", command=check_database)
enroll_btn = tk.Button(root, text="Enrollment", command=create_enroll_wind)
verifi_btn = tk.Button(root, text="Verification", command=create_verifi_wind)
recogn_btn = tk.Button(root, text="Recognition", command=create_recogn_wind)
check_btn.pack()
enroll_btn.pack()
verifi_btn.pack()
recogn_btn.pack()
center(root)
root.mainloop()




























































