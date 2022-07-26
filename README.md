# GaitNet

http://cvlab.cse.msu.edu/pdfs/Zhang_Tran_Yin_Atoum_Liu_Wan_Wang_CVPR2019.pdf

## Prerequisites

- Linux/ Windows
- [Anaconda 3 - Python 3.7 version](https://www.anaconda.com/distribution/#download-section)

## Getting Started

## Datasets

Raw_Dataset
http://cvlab.cse.msu.edu/frontal-view-gaitfvg-database.html

## Installation
- Download codes
> git clone https://github.com/ZiyuanTonyZhang/GaitNet.git

> cd GaitNet/

- Install required libs
> pip install -r requirements.txt

- OR if you use Anaconda
> ------

- Download Dataset
[cropped images](https://drive.google.com/open?id=1nYhhToxjdRp4XFyOIynqVzVxKOcS2FYo)
> Unzip to Data/ , do not change the three folder names

The final folder structure should look like:
- Data
  - SEG-S1
  - SEG-S2
  - SEG-S3
- GaitNet
  - train.py
  - runs
  - ...


## Training and Testing

Since the code is set up with orginal papers defaut hyperparameters, simply run with:

> python train.py

You will be asked which GPU to use, enter 0 if you have only one GPU. If you have multiple GPUs, check their index with

> nvidia-smi

After running train.py, run TensorboardX to visulize training loss curves and synthesized results:

> tensorboard --logdir runs


# GaitNet-CVPR2019

MRCNN: TORCHVISION

IMAGE PROCESSING: PIL, TORCHVISION


1. MASK R-CNN
    1. torchvision
    2. faster (11-12FPS on 1080P)
    3. threading data loader
    4. more effective algorithm to remove redundant data(out of frame)
2. CUDNN
    1. 3 times fater for training

