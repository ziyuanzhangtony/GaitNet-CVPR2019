import random
import os
import numpy as np
import torch
from torchvision import transforms
from scipy.misc import imread
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import spatial
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid,save_image
import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter


# from torch.utils.data import DataLoader
# from torch.autograd import Variable
# from torchvision.utils import make_grid,save_image
# from torch.autograd import Variable
# import math
# import torch.nn.functional as F
# import itertools
# import scipy.io as sio
# from scipy.stats import itemfreq

class FVG(object):

    def __init__(self,
                 train,
                 data_root,
                 sequence_len=30,
                 video_i = range(1, 12+1),
                 image_width=64,
                 image_height=64,
                 seed = 1,
                 ):

        np.random.seed(seed)
        self.data_root = data_root
        self.sequence_len = sequence_len

        if video_i is not []:
            self.video_i = video_i
        else:
            self.video_i = range(1, 12+1)

        self.image_width = image_width
        self.image_height = image_height

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad((50,0)),
            transforms.Resize((image_width,image_width)),
            transforms.ToTensor()
        ])

        subjects = list(range(1, 226+1))
        random.Random(1).shuffle(subjects)
        self.train_subjects = subjects[:136] #136 subjects 60%
        self.test_subjects = subjects[136:] #90 subjects 40%
        print('training_subjects', self.train_subjects, np.sum(self.test_subjects))
        print('testing_subjects',self.test_subjects,np.sum(self.test_subjects))

        self.train = train


    def get_eval_data(self,
                      train,
                      gallery,
                      session1_probe=[],
                      session2_probe=[],
                      ):
        if train == True:
            subjects = self.train_subjects
        else:
            subjects = self.test_subjects

        gt = []

        def read_frames_(si_idx, vi_idx):
            if si_idx in list(range(1,147+1)):
                reading_dir = 'SEG-S1'
            else:
                reading_dir = 'SEG-S2'

            video_in = os.path.join(self.data_root, reading_dir, '%03d_%02d' % (si_idx, vi_idx))
            files = sorted(os.listdir(video_in))
            shape = [len(files), 3, self.image_width, self.image_height]
            data = np.zeros(shape, np.float32)
            for i in range(len(files)):
                try:
                    img = imread(os.path.join(video_in, files[i]))
                    img = self.transform(img)
                    data[i] = img
                except:
                    print('FOUND A BAD IMAGE, SKIPPED')
            return torch.from_numpy(data)

        test_data_glr = []
        test_data_prb = []

        for i,id in enumerate(subjects):
            test_data_glr.append(read_frames_(id, gallery))
            if id in list(range(1, 147 + 1)):
                probes = session1_probe
            else:
                probes = session2_probe
            gt.append(len(probes))
            for j in probes:
                test_data_prb.append(read_frames_(id,j))
            print('Training data:',train,'.Reading',i,'th subject. ')

        test_data_glr = pad_sequences(test_data_glr,maxlen=70,dtype='float32',padding='post')
        test_data_prb = pad_sequences(test_data_prb,maxlen=70,dtype='float32',padding='post')

        return torch.tensor(test_data_glr).permute(1,0,2,3,4),\
               torch.tensor(test_data_prb).permute(1,0,2,3,4),\
               gt

    def get_eval_data_all(self,
                      train,
                      gallery,
                      session1_probe=[],
                      session2_probe=[],
                      session3_probe=[],
                      cross_session=False,
                      ):
        if train == True:
            subjects = self.train_subjects
        else:
            subjects = self.test_subjects

        if cross_session:
            session1_probe = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            session2_probe = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            session3_probe = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

        gt = []

        def read_frames_(reading_dir, si_idx, vi_idx):
            if reading_dir == '':
                if si_idx in list(range(1, 147 + 1)):
                    # if si_idx in [1,2,4,7,8,12,13,17,31,40,48,77]:
                    #     reading_dir = random.choice(['SEG-S1','SEG-S3'])
                    # else:
                    reading_dir = 'SEG-S1'
                else:
                    reading_dir = 'SEG-S2'

            video_in = os.path.join(self.data_root, reading_dir, '%03d_%02d' % (si_idx, vi_idx))
            files = sorted(os.listdir(video_in))
            shape = [len(files), 3, self.image_width, self.image_height]
            data = np.zeros(shape, np.float32)
            # ns = [random.uniform(0.1, 1) for i in range(6)]
            for i in range(len(files)):
                try:
                    img = imread(os.path.join(video_in, files[i]))
                    img = self.transform(img)
                    # img_random_color(img, ns)
                    data[i] = img
                except:
                    print('FOUND A BAD IMAGE, SKIPPED')
            print(reading_dir, si_idx, vi_idx)
            return torch.from_numpy(data)

        test_data_glr = []
        test_data_prb = []

        import time

        for i, id in enumerate(subjects):
            glr_frames = read_frames_('', id, gallery).unsqueeze(0)
            glr_frames = pad_sequences(glr_frames, maxlen=70, dtype='float32', padding='post')
            glr_frames = torch.tensor(glr_frames).permute(1, 0, 2, 3, 4)

            fg_glr = [netE(glr_frames[i].cuda())[1].detach() for i in range(len(glr_frames))]
            fg_glr = torch.stack(fg_glr, 0).view(70, 1, opt.hg_dim)

            st = time.time()
            gf_glr = lstm(fg_glr)[1].detach().cpu().numpy()
            test_data_glr.append(gf_glr)
            t = time.time() - st
            print(t, t / 70)

            if id in list(range(1, 147 + 1)):
                if id in [1, 2, 4, 7, 8, 12, 13, 17, 31, 40, 48, 77]:
                    for j in session1_probe:
                        frames = read_frames_('SEG-S1', id, j)
                        frames = frames.unsqueeze(0)
                        frames = pad_sequences(frames, maxlen=70, dtype='float32', padding='post')
                        frames = torch.tensor(frames).permute(1, 0, 2, 3, 4)
                        fg_prb = [netE(frames[i].cuda())[1].detach() for i in range(len(frames))]
                        fg_prb = torch.stack(fg_prb, 0).view(70, 1, opt.hg_dim)
                        gf_prb = lstm(fg_prb)[1].detach().cpu().numpy()
                        test_data_prb.append(gf_prb)
                    for j in session3_probe:
                        frames = read_frames_('SEG-S3', id, j)
                        frames = frames.unsqueeze(0)
                        frames = pad_sequences(frames, maxlen=70, dtype='float32', padding='post')
                        frames = torch.tensor(frames).permute(1, 0, 2, 3, 4)
                        fg_prb = [netE(frames[i].cuda())[1].detach() for i in range(len(frames))]
                        fg_prb = torch.stack(fg_prb, 0).view(70, 1, opt.hg_dim)
                        gf_prb = lstm(fg_prb)[1].detach().cpu().numpy()
                        test_data_prb.append(gf_prb)
                    gt.append(len(session1_probe) + len(session3_probe))
                    print(i, id, 'SEG-S1&3', len(session1_probe) + len(session3_probe))
                else:
                    for j in session1_probe:
                        frames = read_frames_('SEG-S1', id, j)
                        frames = frames.unsqueeze(0)
                        frames = pad_sequences(frames, maxlen=70, dtype='float32', padding='post')
                        frames = torch.tensor(frames).permute(1, 0, 2, 3, 4)
                        fg_prb = [netE(frames[i].cuda())[1].detach() for i in range(len(frames))]
                        fg_prb = torch.stack(fg_prb, 0).view(70, 1, opt.hg_dim)
                        gf_prb = lstm(fg_prb)[1].detach().cpu().numpy()
                        test_data_prb.append(gf_prb)
                    gt.append(len(session1_probe))
                    print(i, id, 'SEG-S1', len(session1_probe))
            else:
                for j in session1_probe:
                    frames = read_frames_('SEG-S2', id, j)

                    frames = frames.unsqueeze(0)
                    frames = pad_sequences(frames, maxlen=70, dtype='float32', padding='post')
                    frames = torch.tensor(frames).permute(1, 0, 2, 3, 4)
                    fg_prb = [netE(frames[i].cuda())[1].detach() for i in range(len(frames))]
                    fg_prb = torch.stack(fg_prb, 0).view(70, 1, opt.hg_dim)
                    gf_prb = lstm(fg_prb)[1].detach().cpu().numpy()
                    test_data_prb.append(gf_prb)

                gt.append(len(session2_probe))
                print(i, id, 'SEG-S2', len(session2_probe))

        # test_data_glr = pad_sequences(test_data_glr,maxlen=70,dtype='float32',padding='post')
        # test_data_prb = pad_sequences(test_data_prb,maxlen=70,dtype='float32',padding='post')
        test_data_glr = np.stack(test_data_glr, 0)
        test_data_prb = np.stack(test_data_prb, 0)

        return test_data_glr, \
               test_data_prb, \
               gt



    def __getitem__(self, index):
        """
        chose any two videos for the same subject
        :param index:
        :return:
        """
        shape = [self.sequence_len, 3, self.image_width, self.image_height]
        # 30 x 3 x 64 x 64
        def random_si_idx():
            if self.train:
                si_idx = np.random.choice(self.train_subjects)
                labels = self.train_subjects.index(si_idx)
            else:
                si_idx = np.random.choice(self.test_subjects)
                labels = self.test_subjects.index(si_idx)
            return si_idx,labels

        def random_vi_idx(si):


            if si in list(range(1,147+1)):
                if si in [1,2,4,7,8,12,13,17,31,40,48,77]:
                    reading_dir = random.choice(['SEG-S1','SEG-S3'])
                else:
                    reading_dir = 'SEG-S1'
            else:
                reading_dir = 'SEG-S2'

            vi_idx = np.random.choice(self.video_i)

            return reading_dir,vi_idx

        def random_length(dirt,si,length):
            files = sorted(os.listdir(dirt))
            ##########################################################
            if si in list(range(148,226+1)):
                files = files[:-50]
            ##########################################################
            num = len(files)
            if num - length < 2:
                return 0,0,[]
            start = np.random.randint(1, num - length)
            end = start + length
            return start, end, files

        def read_frames(video_in, start, end, files):
            ims = np.zeros(shape, np.float32)
            for i in range(start, end):
                im = imread(os.path.join(video_in, files[i]))
                im = self.transform(im)
                ims[i - start] = im
            return ims

        si, labels = random_si_idx()
        session_dir1, vi1 = random_vi_idx(si)
        session_dir2, vi2 = random_vi_idx(si)
        video_pth1 = os.path.join(self.data_root, session_dir1,'%03d_%02d' % (si, vi1))
        video_pth2 = os.path.join(self.data_root, session_dir2, '%03d_%02d' % (si, vi2))
        start1, end1, files1 = random_length(video_pth1, si,self.sequence_len)
        start2, end2, files2 = random_length(video_pth2, si,self.sequence_len)

        while True:
            if start1==start2==end1==end2:
                session_dir1, vi1 = random_vi_idx(si)
                session_dir2, vi2 = random_vi_idx(si)
                video_pth1 = os.path.join(self.data_root, session_dir1, '%03d_%02d' % (si, vi1))
                video_pth2 = os.path.join(self.data_root, session_dir2, '%03d_%02d' % (si, vi2))
                start1, end1, files1 = random_length(video_pth1, si, self.sequence_len)
                start2, end2, files2 = random_length(video_pth2, si, self.sequence_len)
            else:
                break

        imgs1 = read_frames(video_pth1, start1, end1, files1)
        imgs2 = read_frames(video_pth2, start2, end2, files2)

        return imgs1, imgs2, labels

    def __len__(self):
        if self.train:
            return len(self.train_subjects)*12
        else:
            return len(self.test_subjects)*12


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def process_confusion_matrix(matrix,n_class,gt):
    matrix = np.reshape(matrix,(n_class*sum(gt)))
    def make_labels():
        squre_matrix = np.eye(n_class, n_class)
        matrix = []

        for i in range(n_class):
            for j in range(gt[i]):
                matrix.append(squre_matrix[i])

        matrix = np.asarray(matrix)
        return np.concatenate(matrix)
    labels = make_labels()
    labels = np.reshape(labels,(n_class*sum(gt)))
    fpr, tpr, _ = roc_curve(labels,matrix)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr,roc_auc

def plot_roc(fpr,tpr,roc_auc):
    plt.figure()
    lw = 3
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Alarm Rate/ False Positive Rate')
    plt.ylabel('True Accept Rate/ True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()

def find_idx(fpr,tpr,threthold=[0.01,0.05,0.1],ifround=True):
    outptut = []
    for i in threthold:
        item = fpr[fpr<i+0.005].max()
        idx = np.where(fpr==item)
        val = tpr[idx][-1]
        if ifround:
            val = round(val,2)
        outptut.append(val)
    return outptut

def calculate_cosine_similarity(a, b):
    score = 1 - spatial.distance.cosine(a, b)
    return score

def calculate_identication_rate_single(glrs,aprb,trueid,rank=1):
    scores = []
    for i in glrs:
        scores.append(calculate_cosine_similarity(i,aprb))
    max_val = max(scores)
    max_idx = scores.index(max_val)

    right,predicted = trueid,max_idx
    print(right,predicted )


    if max_idx in trueid:
        return 1,[right,predicted]
    else:
        return 0,[right,predicted]

def adjust_white_balance(x):
    x = x.cpu()
    avgR = np.average(x[0,:,:])
    avgG = np.average(x[1,:,:])
    avgB = np.average(x[2,:,:])

    avg = (avgB + avgG + avgR) / 3

    x[0,:,:] = np.minimum(x[0] * (avg / avgR), 1)
    x[1,:,:] = np.minimum(x[1] * (avg / avgG), 1)
    x[2,:,:] = np.minimum(x[2] * (avg / avgB), 1)

    return x

def adjust_(x):
    x = transforms.functional.adjust_brightness(x, 1.5)
    x = transforms.functional.adjust_contrast(x, 1.5)
    return x

#################################################################################################################

class vgg_layer(nn.Module):
    def __init__(self, nin, nout):
        super(vgg_layer, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 3, 1, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2)
                )

    def forward(self, input):
        return self.main(input)
class dcgan_conv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2),
                )

    def forward(self, input):
        return self.main(input)
class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_upconv, self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(nin, nout, 4, 2, 1,),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2),
                )

    def forward(self, input):
        return self.main(input)
class encoder(nn.Module):
    def __init__(self, nc=3):
        super(encoder, self).__init__()
        self.em_dim = opt.em_dim
        nf = 64
        self.main = nn.Sequential(
            dcgan_conv(nc, nf),
            vgg_layer(nf, nf ),
            dcgan_conv(nf, nf * 2),
            vgg_layer(nf*2, nf*2),
            dcgan_conv(nf * 2, nf * 4),
            vgg_layer(nf*4, nf*4),
            dcgan_conv(nf * 4, nf * 8),
            vgg_layer(nf*8, nf*8),

            nn.Conv2d(nf * 8, self.em_dim, 4, 1, 0),
            nn.BatchNorm2d(self.em_dim),
        )

        self.ha_fc = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(self.em_dim, self.em_dim // 2),
            nn.BatchNorm1d(self.em_dim // 2),

            nn.LeakyReLU(),
            nn.Linear(self.em_dim // 2, self.em_dim // 2),
            nn.BatchNorm1d(self.em_dim // 2),

            nn.LeakyReLU(),
            nn.Linear(self.em_dim // 2, opt.ha_dim),
            nn.BatchNorm1d(opt.ha_dim)
        )

        self.hg_fc = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(self.em_dim,self.em_dim//2),
            nn.BatchNorm1d(self.em_dim//2),

            nn.LeakyReLU(),
            nn.Linear(self.em_dim // 2, self.em_dim // 2),
            nn.BatchNorm1d(self.em_dim // 2),

            nn.LeakyReLU(),
            nn.Linear(self.em_dim//2, opt.hg_dim),
            nn.BatchNorm1d(opt.hg_dim)
        )


    def forward(self, input):
        embedding = self.main(input).view(-1,self.em_dim)
        ha,hg = torch.split(embedding, [opt.ha_dim, opt.hg_dim], dim=1)
        return ha,hg,embedding
class decoder(nn.Module):
    def __init__(self, nc=3):
        super(decoder, self).__init__()
        nf = 64
        self.em_dim = opt.em_dim
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.em_dim, nf * 8, 4, 1, 0),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2),
            vgg_layer(nf*8, nf*8),

            dcgan_upconv(nf * 8, nf * 4),
            vgg_layer(nf * 4, nf * 4),

            dcgan_upconv(nf * 4, nf * 2),
            vgg_layer(nf * 2, nf * 2),

            dcgan_upconv(nf * 2, nf),
            vgg_layer(nf, nf),

            nn.ConvTranspose2d(nf, nc, 4, 2, 1),
            nn.Sigmoid()
                # because image pixels are from 0-1, 0-255
                )

    def forward(self,ha,hg):
        return self.main(torch.cat([ha, hg], 1).view(-1,self.em_dim,1,1))
class lstm(nn.Module):
    def __init__(self, hidden_dim=128, tagset_size=136):
        super(lstm, self).__init__()
        self.source_dim = opt.hg_dim
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size
        self.lens = 0
        self.lstm = nn.LSTM(self.source_dim, hidden_dim, 3)
        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),

        )
        self.main = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(self.hidden_dim, tagset_size),
            nn.BatchNorm1d(tagset_size),
        )
    def forward(self, batch):
        lens = batch.shape[0]
        lstm_out, _ = self.lstm(batch.view(lens,-1,self.source_dim))
        lstm_out_test = self.fc1(torch.mean(lstm_out.view(lens,-1,self.hidden_dim),0))
        # lstm_out_test = self.fc1(lstm_out.view(lens,-1,self.hidden_dim)[-1])
        # lstm_out_test = torch.mean(batch.view(lens,-1,64),0)
        lstm_out_train = self.main(lstm_out_test).view(-1, self.tagset_size)
        return lstm_out_train,lstm_out_test,lstm_out

#################################################################################################################
# HYPER PARAMETERS INITIALIZING
parser = argparse.ArgumentParser()
gpu_num = int(input('Enter the gpu for this experiment:'))
parser.add_argument('--gpu', type=int, default=gpu_num)
parser.add_argument('--siter', type=int, default=0, help='number of itr to start with')
parser.add_argument('--lr', default=0.00001, type=float, help='learning rate')
parser.add_argument('--data_root',
                     default='../Data/',
                     help='root directory for data')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--em_dim', type=int, default=320, help='size of the pose vector')
parser.add_argument('--ha_dim', type=int, default=288, help='size of the appearance vector')
parser.add_argument('--hg_dim', type=int, default=32, help='size of the gait vector')
parser.add_argument('--image_width', type=int, default=64, help='the width of the input image to network')
parser.add_argument('--image_height', type=int, default=64, help='the height of the input image to network')
parser.add_argument('--max_step', type=int, default=20, help='maximum distance between frames')
parser.add_argument('--data_threads', type=int, default=2, help='number of parallel data loading threads')
parser.add_argument('--normalize', action='store_true', help='if true, normalize pose vector')

import datetime
time_now = str(datetime.datetime.now())
# time_now = '2018-11-07 11:13:23.782672'
parser.add_argument('--signature', default=time_now)
parser.add_argument('--savedir', default='./runs')
opt = parser.parse_args()
print(opt)
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
os.makedirs('%s/analogy/%s'%(opt.savedir,opt.signature), exist_ok=True)
os.makedirs('%s/modules/%s'%(opt.savedir,opt.signature), exist_ok=True)
#################################################################################################################
# MODEL PROCESS
netE = encoder()
netD = decoder()
lstm = lstm()
netE.apply(init_weights)
netD.apply(init_weights)
lstm.apply(init_weights)
if opt.siter is not 0:
    checkpoint = torch.load('%s/modules/%s/%d.pickle' % (opt.savedir,opt.signature, opt.siter))
    netE.load_state_dict(checkpoint['netE'])
    netD.load_state_dict(checkpoint['netD'])
    lstm.load_state_dict(checkpoint['lstm'])
    print('model loadinged successfully')

# optimizerE = optim.Adam(netE.parameters(), lr=opt.lr, betas=(0.9, 0.999),weight_decay=0.001)
# optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.9, 0.999),weight_decay=0.001)
# optimizerLstm = optim.Adam(lstm.parameters(), lr=opt.lr, betas=(0.9, 0.999),weight_decay=0.001)
# optimizerClfer = optim.Adam(clfer.parameters(), lr=opt.lr, betas=(0.9, 0.999),weight_decay=0.001)

optimizerE = optim.Adam(netE.parameters(), lr=opt.lr, betas=(0.9, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.9, 0.999))
optimizerLstm = optim.Adam(lstm.parameters(), lr=opt.lr, betas=(0.9, 0.999))

mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()
cse_loss = nn.CrossEntropyLoss()
trp_loss = nn.TripletMarginLoss(margin=2.0)
netE.cuda()
netD.cuda()
lstm.cuda()
mse_loss.cuda()
bce_loss.cuda()
cse_loss.cuda()
trp_loss.cuda()

# #################################################################################################################
# DATASET PREPARATION
def get_training_batch(data_loader):
    while True:
        for sequence in data_loader:
            batch = sequence[0].cuda(),sequence[1].cuda(),sequence[2].cuda()
            yield batch

train_data1 = FVG(
    train=True,
    data_root=opt.data_root,
    sequence_len=opt.max_step,
)

train_loader1 = DataLoader(train_data1,
                          num_workers=opt.data_threads,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)

training_batch_generator1 = get_training_batch(train_loader1)
test_data = FVG(
    train=False,
    data_root=opt.data_root,
    sequence_len=opt.max_step,
)
test_loader = DataLoader(test_data,
                         num_workers=opt.data_threads,
                         batch_size=opt.batch_size,
                         shuffle=True,
                         drop_last=True,
                         pin_memory=True)
testing_batch_generator = get_training_batch(test_loader)

#################################################################################################################

def make_analogy_inter_subs(x):
    # (B, L, C, H, W)
    # (   B, C, H, W)
    # netE.eval()
    # netD.eval()

    def rand_idx():
        return np.random.randint(0, opt.batch_size)

    def rand_step():
        # return np.random.randint(0, opt.max_step)
        return 0

    none = torch.zeros([1, 3, 64, 64]).cuda()
    x_gs = torch.stack([i for i in [x[0][rand_step()], x[1][rand_step()], x[2][rand_step()], x[3][rand_step()]]]).cuda()
    h_gs = netE(x_gs)[1]
    # h_gs = torch.zeros(4,32).cuda()

    x_as = torch.stack([x[i][rand_step()] for i in [1, 2, 11]]).cuda()
    # x_as = torch.stack([x[i][0] for i in [2, 2, 2, 2, 2]]).cuda()

    h_as = netE(x_as)[0]
    # h_as = torch.zeros(3,288).cuda()

    gene = [netD(torch.stack([i] * 4).cuda(), h_gs) for i in h_as]
    row0 = torch.cat([none, x_gs])
    rows = [torch.cat([e.unsqueeze(0), gene[i]]) for i, e in enumerate(x_as)]
    to_plot = torch.cat([row0] + rows)

    img = make_grid(to_plot,5)
    return img

def make_analogy_intal_subs(nm,cl,idx):
    # (B, L, C, H, W)
    # (   B, C, H, W)
    # netE.eval()
    # netD.eval()

    def rand_idx():
        return 0

    def rand_step():
        # return np.random.randint(0, opt.max_step)
        return 0

    none = torch.zeros([1, 3, 64, 64]).cuda()
    x_gs = torch.stack([i for i in [nm[idx][0], nm[idx][4], nm[idx][8], nm[idx][15]]]).cuda()
    h_gs = netE(x_gs)[1]
    # h_gs = torch.zeros(4,32).cuda()

    x_as = torch.stack( [nm[idx][rand_step()],cl[idx][rand_step()]]).cuda()
    # x_as = torch.stack([x[i][0] for i in [2, 2, 2, 2, 2]]).cuda()

    h_as = netE(x_as)[0]
    # h_as = torch.zeros(2,288).cuda()

    gene = [netD(torch.stack([i] * 4).cuda(), h_gs) for i in h_as]
    row0 = torch.cat([none, x_gs])
    rows = [torch.cat([e.unsqueeze(0), gene[i]]) for i, e in enumerate(x_as)]
    to_plot = torch.cat([row0] + rows)

    img = make_grid(to_plot,5)
    return img

def plot_anology(data1,data2,epoch):
    anology1 = make_analogy_inter_subs(data1)
    anology2 = make_analogy_inter_subs(data2)
    all = torch.cat([anology1,anology2],dim=1)
    all = adjust_white_balance(all.detach())
    writer.add_image('inter_sub', all, epoch)

    anology1 = make_analogy_intal_subs(data1,data2,8)
    anology2 = make_analogy_intal_subs(data1,data2,9)
    all = torch.cat([anology1, anology2], dim=1)
    all = adjust_white_balance(all.detach())
    writer.add_image('intral_sub', all, epoch)

def eval_lstm_roc(glr, prb, gt, n_test  = 90):

    def calculate_cosine_similarity(a, b):
        score = 1 - spatial.distance.cosine(a, b)
        return score

    hp_glr = [netE(glr[i].cuda())[1].detach() for i in range(len(glr))]
    # hp_glr = [torch.cat([netEc(glr[0])[1].detach(), netEp(glr[i])[0].detach()], 1)
    #           for i in range(len(glr))]
    hp_glr = torch.stack(hp_glr, 0).view(len(hp_glr), n_test, opt.hg_dim)
    glr_vec = lstm(hp_glr)[1].detach().cpu().numpy()
    # glr_vec = torch.mean(hp_glr, 0).detach().cpu().numpy()

    hp_prb = [netE(prb[i].cuda())[1].detach() for i in range(len(prb))]
    # hp_prb = [torch.cat([netEc(prb[0])[1].detach(), netEp(prb[i])[0].detach()], 1)
    #           for i in range(len(prb))]

    hp_prb = torch.stack(hp_prb, 0).view(len(hp_prb), -1, opt.hg_dim)
    prb_vec = lstm(hp_prb)[1].detach().cpu().numpy()
    # prb_vec = torch.mean(hp_prb, 0).detach().cpu().numpy()

    obj_arr = np.zeros((len(prb_vec), n_test), dtype=np.float32)
    for i in range(n_test):
        for j in range(len(prb_vec)):
            cs = calculate_cosine_similarity(glr_vec[i:i + 1, :],
                                             prb_vec[j:j + 1, :])
            obj_arr[j, i] = cs
    fpr, tpr,roc_auc = process_confusion_matrix(obj_arr,n_test,gt)
    return find_idx(fpr, tpr)

def eval_lstm_cmc(glr, prb):
    pb_vecs = []
    gr_vecs = []
    for pb in prb:
        fg_pb = [netE(pb[i].cuda())[1].detach() for i in range(len(pb))]
        fg_pb = torch.stack(fg_pb, 0).view(len(fg_pb), -1, opt.hg_dim)
        pb_vec = lstm(fg_pb)[1].detach().cpu().numpy()
        pb_vecs.append(pb_vec)

    for gr in glr:
        fg_gr = [netE(gr[i].cuda())[1].detach() for i in range(len(gr))]
        fg_gr = torch.stack(fg_gr, 0).view(len(fg_gr), -1, opt.hg_dim)
        gr_vec = lstm(fg_gr)[1].detach().cpu().numpy()
        gr_vecs.append(gr_vec)

    scores_all = []
    for pb_idx,pv in enumerate(pb_vecs):
        scores_this_pv = []
        for gv_idx,gv in enumerate(gr_vecs):
            if opt.glr_views[gv_idx] != opt.prb_views[pb_idx]:
                score = []
                for i in range(len(pv)):
                    id = i
                    id_range = list(range(id,id+1))
                    score.append(calculate_identication_rate_single(gv, pv[i], id_range)[0])
                score = sum(score) / float(len(score))
                scores_this_pv.append(score)
        scores_this_pv = sum(scores_this_pv) / float(len(scores_this_pv))
        scores_all.append(scores_this_pv)
    return scores_all

def write_tfboard(vals,itr,name):
    for idx,item in enumerate(vals):
        writer.add_scalar('data/%s%d'%(name,idx), item, itr)

#################################################################################################################
# TRAINING FUNCTION DEFINE
def train_main(Xn, Xc, Xmx, l):
    Xn, Xc, Xmx = Xn.transpose(0, 1), Xc.transpose(0, 1), Xmx.transpose(0, 1)
    hgs_n = []
    hgs_c = []

    rec_loss = 0
    for i in range(5, len(Xn)):
        netE.zero_grad()
        netD.zero_grad()
        lstm.zero_grad()
        rdm = torch.LongTensor(1).random_(5, len(Xn))[0]
        # ------------------------------------------------------------------
        xmx0, xmx1 = Xmx[rdm], Xmx[i]
        hamx0, hgmx0, emmx0 = netE(xmx0)
        hamx1, hgmx1, emmx1 = netE(xmx1)
        # ------------------------------------------------------------------
        xmx1_ = netD(hamx0, hgmx1)
        rec_loss += mse_loss(xmx1_, xmx1)
        # ------------------------------------------------------------------
        xn1, xc1 = Xn[i], Xc[i]
        han1, hgn1, _ = netE(xn1)
        hac1, hgc1, _ = netE(xc1)

        hgs_n.append(hgn1)
        hgs_c.append(hgc1)

    hgs_n = torch.stack(hgs_n)
    hgs_n = torch.mean(hgs_n, 0)

    hgs_c = torch.stack(hgs_c)
    hgs_c = torch.mean(hgs_c, 0)


    gait_sim_loss = mse_loss(hgs_n, hgs_c) / 100000

    loss = rec_loss + gait_sim_loss

    loss.backward()
    optimizerE.step()
    optimizerD.step()

    return [rec_loss.data.cpu().numpy(),
            gait_sim_loss.data.cpu().numpy()]


def train_lstm(x_mx,l):
    x_mx = x_mx.transpose(0, 1)
    id_incre_loss=0
    hgs_mx = []
    for i in range(0, len(x_mx)):
        netE.zero_grad()
        netD.zero_grad()
        lstm.zero_grad()
        factor = (i/5)**2/10

        xmx = x_mx[i]
        hgs_mx.append(netE(xmx)[1])
        lstm_out_mx = lstm(torch.stack(hgs_mx))[0]
        id_incre_loss += cse_loss(lstm_out_mx,Variable(l))*factor

    id_incre_loss /= opt.max_step
    id_incre_loss *= 0.1

    los = id_incre_loss
    los.backward()
    optimizerLstm.step()
    optimizerE.step()
    return [los.data.cpu().numpy()]

#################################################################################################################
# FUN TRAINING TIME !
# train_eval = test_data.get_eval_data(True)
proto1 = test_data.get_eval_data(False,2,session1_probe=list(range(4, 9 + 1)),session2_probe=list(range(4, 6 + 1)))
proto2 = test_data.get_eval_data(False,2,session1_probe=list(range(10, 12 + 1)))
proto3 = test_data.get_eval_data(False,2,session2_probe=list(range(7, 9 + 1)))
proto4 = test_data.get_eval_data(False,2,session2_probe=list(range(10, 12 + 1)))
# proto5 = test_data.get_eval_data_all(False,2,cross_session=True)
writer = SummaryWriter('%s/logs/%s'%(opt.savedir,opt.signature))
itr = opt.siter
while True:
    netE.train()
    netD.train()
    lstm.train()

    im_cond1, im_cond2,lb = next(training_batch_generator1)

    losses1 = train_main(im_cond1, im_cond2, im_cond1,lb)
    write_tfboard(losses1,itr,name='EDLoss')

    losses3 = train_lstm(im_cond1,lb)
    write_tfboard(losses3, itr, name='LstmLoss')
    print(itr)

    # ----------------EVAL()--------------------
    if itr % 50 == 0:
        netD.eval()
        netE.eval()
        lstm.eval()
        scores1 = eval_lstm_roc(proto1[0], proto1[1], proto1[2])
        scores2 = eval_lstm_roc(proto2[0], proto2[1], proto2[2])
        scores3 = eval_lstm_roc(proto3[0], proto3[1], proto3[2])
        scores4 = eval_lstm_roc(proto4[0], proto4[1], proto4[2])
        # scores5 = eval_lstm_roc(proto5[0], proto5[1], proto5[2])

        write_tfboard(scores1[:2], itr, name='WS')
        write_tfboard(scores2[:2], itr, name='CB-OLD')
        write_tfboard(scores3[:2], itr, name='CL')
        write_tfboard(scores4[:2], itr, name='CBG')
        # write_tfboard(scores5[:2], itr, name='ALL')


        im_cond1_, im_cond2_, _= next(testing_batch_generator)
        plot_anology(im_cond1_,im_cond2_,itr)

        # ----------------SAVE MODEL--------------------
    if itr % 1000 == 0:
        torch.save({
            'netD': netD.state_dict(),
            'netE': netE.state_dict(),
            'lstm':lstm.state_dict(),
            },
            '%s/modules/%s/%d.pickle'%(opt.savedir,opt.signature,itr),)

    itr+=1


