import torchvision
import torch
import PIL
import time
import os
from torch.utils.data import DataLoader
import numpy as np
from segmentation.mrcnn_resnet50_fpn import MRCNN
from torchvision import transforms
# ========================================================================
in_data_root = '/media/tony/MyBook-MSU-CVLAB/FVG/RAW/'
seg_out_data_root = '/home/tony/Research/NEW-MRCNN___/SEG/'
sil_out_data_root = '/home/tony/Research/NEW-MRCNN___/SIL/'
crop_out_data_root = '/home/tony/Research/NEW-MRCNN___/CROP/'
# out_data_root = '/media/tony/MyBook-MSU-CVLAB/FVG/SEG/'

if_gpu = True
torch.cuda.set_device(0)
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

mrcnn = MRCNN(is_gpu=if_gpu)

number_image_per_batch = 10

threshold = 0.9

fvg_structure = {
    'session1': list(range(1,147+1)),
    'session2': list(range(148,226+1)),
    'session3': [1,2,4,7,8,12,13,17,31,40,48,77],
}

# ========================================================================

class FVG(object):

    def __init__(self,
                 folder_path,
                 frame_height=1080,
                 frame_width=1920,
                 ):
        self.folder_path = folder_path
        self.frame_names = sorted(os.listdir(self.folder_path))
        self.frame_names = [f for f in self.frame_names if f.endswith('.png')]
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.transform = transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.Resize((270, 480)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        try:
            frame = PIL.Image.open(os.path.join(self.folder_path,
                                        self.frame_names[index]))
            frame = self.transform(frame)
        except:
            frame = torch.zeros(3,self.frame_height,self.frame_width)
        return self.frame_names[index],frame


    def __len__(self):
        return len(self.frame_names)

def pre_process_batch(batch):
    new_format = []
    for i in batch:
        new_format.append(i)
    return new_format

def fvg_save_processed_frames(in_path, seg_out_path, sil_out_path, crop_out_path):
    mrcnn.reset()
    try:
        os.makedirs(seg_out_path)
        os.makedirs(sil_out_path)
        os.makedirs(crop_out_path)
    except:
        pass
    fvg_instance = FVG(folder_path=os.path.join(in_path))

    loader = DataLoader(fvg_instance,
                               num_workers=8,
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
            segmentations, silhouettes, crops = mrcnn.process_batch(batch_frame, 0.9, 1, 10, 'FVG')
            print( len(batch_frame)/(time.time()-start) )

            for segmentation,silhouette,crop,file_name in zip(segmentations, silhouettes, crops, batch_frame_names):
                try:
                    torchvision.utils.save_image(segmentation, os.path.join(seg_out_path,file_name))
                    torchvision.utils.save_image(silhouette, os.path.join(sil_out_path, file_name))
                    torchvision.utils.save_image(crop, os.path.join(crop_out_path, file_name))
                except:
                    print("save error, could be empty")

for session, sub_ids in fvg_structure.items():
    for sub_id in sub_ids:
        for vi_idx in range(1,12+1):
            print('{:03d}_{:02d}'.format(sub_id, vi_idx))
            in_folder_name = os.path.join(in_data_root,session,'{:03d}_{:02d}'.format(sub_id, vi_idx))
            seg_out_folder_name = os.path.join(seg_out_data_root,session,'{:03d}_{:02d}'.format(sub_id, vi_idx))
            sil_out_folder_name = os.path.join(sil_out_data_root, session, '{:03d}_{:02d}'.format(sub_id, vi_idx))
            crop_out_folder_name = os.path.join(crop_out_data_root, session, '{:03d}_{:02d}'.format(sub_id, vi_idx))
            if os.path.exists(seg_out_folder_name):
                continue
            else:
                fvg_save_processed_frames(in_folder_name,
                                          seg_out_folder_name,
                                          sil_out_folder_name,
                                          crop_out_folder_name)
