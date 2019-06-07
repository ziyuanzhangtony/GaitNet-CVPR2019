import torchvision
import torch
import PIL
import time
import os
from torch.utils.data import DataLoader
import numpy as np
from segmentation.mrcnn_resnet50_fpn import MRCNN
from torchvision import transforms
import cv2
# ========================================================================
in_data_root = '/home/tony/Documents/CASIA-B/videos'
seg_out_data_root = '/home/tony/Documents/CASIA-B-/SEG/'
sil_out_data_root = '/home/tony/Documents/CASIA-B-/SIL/'
crop_out_data_root = '/home/tony/Documents/CASIA-B-/CROP/'
# out_data_root = '/media/tony/MyBook-MSU-CVLAB/FVG/SEG/'

if_gpu = True
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

mrcnn = MRCNN(is_gpu=if_gpu)

number_image_per_batch = 10

threshold = 0.9

cb_structure = {
    'nm': list(range(1,6+1)),
    'cl': list(range(1,2+1)),
    'bg': list(range(1,2+1)),
}

# ========================================================================

class CASIAB(object):

    def __init__(self,
                 file_path,
                 frame_height=240,
                 frame_width=320,
                 ):
        self.file_path = file_path
        self.cap = cv2.VideoCapture(self.file_path)
        self.video_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.transform = transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.Resize((270, 480)),
            transforms.ToTensor()
        ])
        self.count = 0
        self.success = True

    def __getitem__(self, index):


        self.success, frame = self.cap.read()
        self.count += 1
        if not self.success:
            frame = torch.zeros(3, self.frame_height, self.frame_width)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.transform(frame)
        return "{:04d}.png".format(self.count), frame


    def __len__(self):
        return self.video_length

def pre_process_batch(batch):
    new_format = []
    for i in batch:
        new_format.append(i)
    return new_format

def fvg_save_processed_frames(in_path, seg_out_path, sil_out_path, crop_out_path):
    mrcnn.reset()
    # try:
    os.makedirs(seg_out_path)
    os.makedirs(sil_out_path)
    os.makedirs(crop_out_path)
    # except:
    #     pass
    fvg_instance = CASIAB(file_path=os.path.join(in_path))

    loader = DataLoader(fvg_instance,
                               num_workers=0,
                               batch_size=number_image_per_batch,
                               shuffle=False,
                               drop_last=False,
                               pin_memory=True)

    with torch.no_grad():
        for batch_frame_names, batch_frame in loader:
            if if_gpu:
                batch_frame = batch_frame.cuda()
            batch_frame = pre_process_batch(batch_frame)
            start = time.time()
            segmentations, silhouettes, crops = mrcnn.process_batch(batch_frame, 0.9, 1, 3, 'CB')
            print( len(batch_frame)/(time.time()-start) )

            for segmentation,silhouette,crop,file_name in zip(segmentations, silhouettes, crops, batch_frame_names):
                try:
                    torchvision.utils.save_image(segmentation, os.path.join(seg_out_path,file_name))
                    torchvision.utils.save_image(silhouette, os.path.join(sil_out_path, file_name))
                    torchvision.utils.save_image(crop, os.path.join(crop_out_path, file_name))
                except:
                    print("save error, could be empty")

for sub_id in range(1,124+1):
    for cond,vi_idxs in cb_structure.items():
        for vi_idx in vi_idxs:
            for view_angle in range(0,180+1,18):
                video_name = '{:03d}-{:s}-{:02d}-{:03d}.avi'.format(sub_id, cond, vi_idx, view_angle)
                print(video_name)
                video_path = os.path.join(in_data_root,video_name)
                seg_out_folder_name = os.path.join(seg_out_data_root,video_name.split('.')[0])
                sil_out_folder_name = os.path.join(sil_out_data_root, video_name.split('.')[0])
                crop_out_folder_name = os.path.join(crop_out_data_root, video_name.split('.')[0])
                if os.path.exists(seg_out_folder_name):
                    continue
                else:
                    fvg_save_processed_frames(video_path,
                                              seg_out_folder_name,
                                              sil_out_folder_name,
                                              crop_out_folder_name)
