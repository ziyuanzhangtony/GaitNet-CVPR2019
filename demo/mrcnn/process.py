# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import sys
import argparse
# from maskrcnn_benchmark.config import cfg
import cv2
# from utils.helper import image_resize
import time
from segmentation.mrcnn_resnet50_fpn import MRCNN
from torchvision import transforms
import torch
import torchvision

mrcnn = MRCNN(is_gpu=True)

transform = transforms.Compose([
    transforms.ToTensor()
])

def get_seg_batch(frames, batch_size):
    # coco_demo.is_continue = True #important
    pbbox = tuple()
    segs = []
    time_start = time.time()
    idx = 0
    while True:
        if idx >= len(frames):
            break
        # if len(pbbox) == 0: # process the first WHOLE frame
        #     sub_frames = frames[idx:idx+1]
        #     print("MRCNN Progress:{} / {}".format(idx+1, len(frames)))
        #     idx+=1
        # else:
        if idx+batch_size >= len(frames):
            sub_frames = frames[idx:len(frames)]
            print("MRCNN Progress:{}:{} / {}".format(idx, len(frames), len(frames)))
        else:
            sub_frames = frames[idx:idx+batch_size]
            print("MRCNN Progress:{}:{} / {}".format(idx, idx + batch_size, len(frames)))
        idx+=batch_size
        time_test_s = time.time()
        with torch.no_grad():
            new_format = []
            for frame in sub_frames:
                frame = transform(frame)
                frame = frame.cuda()
                new_format.append(frame)
            time_test1_s = time.time()
            composites, silhouettes, crops = mrcnn.process_batch(new_format, 0.9, 1, 10, 'FVG')
            print("t1:" + str(time.time()-time_test1_s))
            for composite in composites:
                if len(composite) > 0:
                    segs.append(composite.cpu())
        print("t:" + str(time.time()-time_test_s))

    duration = round(time.time() - time_start,2)
    print("d:" + str(duration))
    print("MRCNN TIME:{} seconds".format(duration))
    print("MRCNN FPS:", round(len(frames) / duration,2))
    print()
    return segs