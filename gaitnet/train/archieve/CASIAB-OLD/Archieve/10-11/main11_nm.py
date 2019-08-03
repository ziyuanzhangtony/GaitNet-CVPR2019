import random
import os
import numpy as np
import torch
from torchvision import transforms
from scipy.misc import imread
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision.utils import make_grid,save_image
from sklearn.metrics import roc_curve, auc
from scipy import spatial
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import make_grid,save_image
import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
import itertools
import scipy.io as sio
from tensorboardX import SummaryWriter
from scipy.stats import itemfreq

class CASIAB(object):

    def __init__(self,
                 train,
                 data_root,
                 sequence_len=30,
                 view_i=[90],
                 image_width=64,
                 image_height=64,
                 seed=1,
                 n_train=74,
                 ):

        np.random.seed(seed)
        self.train = train
        self.data_root = data_root
        self.sequence_len = sequence_len
        self.view_i = view_i
        self.image_width = image_width
        self.image_height = image_height
        self.n_train = n_train


        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad((50, 0)),
            transforms.Resize((image_width, image_width)),
            transforms.ToTensor(),
        ])

        subjects = list(range(1, 124 + 1))
        self.train_subjects = subjects[:n_train]
        self.test_subjects = subjects[n_train:]
        # self.test_subjects = subjects[:n_train]
        print('training_subjects', self.train_subjects, np.sum(self.train_subjects), len(self.train_subjects))
        print('testing_subjects', self.test_subjects, np.sum(self.test_subjects), len(self.test_subjects))

    def img_random_color(self,im, ns):
        im[0, :, :] *= ns[0]
        im[1, :, :] *= ns[1]
        im[2, :, :] *= ns[2]
        return im

    def __getitem__(self, index):
        shape = [self.sequence_len, 3, self.image_width, self.image_height]
        imgs_nm = np.zeros(shape, np.float32)
        imgs_cl = np.zeros(shape, np.float32)
        ns = np.random.uniform(0,1,size=3)

        def random_chose():
            if self.train:
                si_idx = np.random.choice(self.train_subjects)
                labels = self.train_subjects.index(si_idx)
            else:
                si_idx = np.random.choice(self.test_subjects)
                labels = self.test_subjects.index(si_idx)
            vi_idx = np.random.choice(self.view_i)
            return si_idx, vi_idx, labels

        def random_length(dirt, length):
            files = sorted(os.listdir(dirt))
            num = len(files)
            if num - length < 2:
                return 0, 0, []
            start = np.random.randint(1, num - length)
            end = start + length
            return start, end, files

        def read_frames(video_in, start, end, files):
            ims = np.zeros(shape, np.float32)

            for i in range(start, end):
                im = imread(os.path.join(video_in, files[i]))
                im = self.transform(im)
                # im = self.img_random_color(im,ns)
                ims[i - start] = im
            return ims

        si, vi, labels = random_chose()
        video_in_nm = os.path.join(self.data_root,
                                   '%03d-%1d-%02d-%03d' % (si, 1, np.random.choice(list(range(1, 6 + 1))), vi))
        start_nm, end_nm, files_nm = random_length(video_in_nm, self.sequence_len)

        video_in_cl = os.path.join(self.data_root,
                                   '%03d-%1d-%02d-%03d' % (si, 1, np.random.choice(list(range(1, 2 + 1))), vi))
        start_cl, end_cl, files_cl = random_length(video_in_cl, self.sequence_len)

        while True:
            if start_nm == end_nm == 0 or start_cl == end_cl == 0:
                si, vi, labels = random_chose()
                video_in_nm = os.path.join(self.data_root,
                                           '%03d-%1d-%02d-%03d' % (si, 1, np.random.choice(list(range(1, 6 + 1))), vi))
                start_nm, end_nm, files_nm = random_length(video_in_nm, self.sequence_len)

                video_in_cl = os.path.join(self.data_root,
                                           '%03d-%1d-%02d-%03d' % (si, 1, np.random.choice(list(range(1, 2 + 1))), vi))
                start_cl, end_cl, files_cl = random_length(video_in_cl, self.sequence_len)
            else:
                break

        TF = random.choice([True, False])
        if TF:
            video_in_mx, start_mx, end_mx, files_mx = \
                video_in_nm, start_nm, end_nm, files_nm
        else:
            video_in_mx, start_mx, end_mx, files_mx = \
                video_in_cl, start_cl, end_cl, files_cl

        imgs_nm = read_frames(video_in_nm, start_nm, end_nm, files_nm)
        imgs_cl = read_frames(video_in_cl, start_cl, end_cl, files_cl)
        imgs_mx = read_frames(video_in_mx, start_mx, end_mx, files_mx)
        return imgs_nm, imgs_cl,imgs_mx, labels

    def get_eval_data(self,train=False):
        test_data_glr_nm = []
        test_data_prb_cnd = []

        def read_frames_(si, ci, sei, vi):
            video_in = os.path.join(self.data_root, '%03d-%1d-%02d-%03d' % (si, ci, sei, vi))
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
            return data

        if train:
            data_type = self.train_subjects
        else:
            data_type = self.test_subjects

        for i, id in enumerate(data_type):
            for ci in range(1, 3 + 1):
                if ci == 1:
                    seqis = range(1, 6 + 1)
                else:
                    seqis = range(1, 2 + 1)
                for seqi in seqis:
                    for vi in range(0, 180 + 1, 18):
                        if ci == 1:
                            if seqi in [1]:
                                if vi == 90:
                                    test_data_glr_nm.append(read_frames_(id, ci, seqi, vi))
                        if ci == 2:
                            if seqi in [1]:
                                if vi == 90:
                                    test_data_prb_cnd.append(read_frames_(id, ci, seqi, vi))
            print('Loading subject:', i)

        test_data_glr = pad_sequences(test_data_glr_nm, maxlen=70, dtype='float32', padding='post')
        test_data_prb = pad_sequences(test_data_prb_cnd, maxlen=70, dtype='float32', padding='post')

        return torch.tensor(test_data_glr).permute(1, 0, 2, 3, 4), \
               torch.tensor(test_data_prb).permute(1, 0, 2, 3, 4),

    def __len__(self):
        if self.train:
            return len(self.train_subjects)*110
        else:
            return len(self.test_subjects)*110

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def process_confusion_matrix(matrix,n_class,n_sample):
    matrix = np.reshape(matrix,(n_class*n_class*n_sample))
    def make_labels():
        matrix = [np.eye(n_class, n_class)[j] for j in range(n_class) for _ in range(n_sample)]
        return np.concatenate(matrix)
    labels = make_labels()
    labels = np.reshape(labels,(n_class*n_class*n_sample))
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


    if max_idx in trueid:
        return 1,[right,predicted]
    else:
        return 0,[right,predicted]

def adjust_white_balance(x):

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

class classification(nn.Module):
    def __init__(self):
        super(classification, self).__init__()
        self.out_dim = opt.num_train

        self.main_em = nn.Sequential(
            nn.Linear(opt.em_dim, opt.em_dim*2),
            nn.BatchNorm1d(opt.em_dim*2),
            nn.LeakyReLU(),

            nn.Linear(opt.em_dim*2, opt.em_dim*4),
            nn.BatchNorm1d(opt.em_dim*4),
            nn.LeakyReLU(),

            nn.Linear(opt.em_dim*4,self.out_dim),
            nn.BatchNorm1d(self.out_dim)

        )

        self.main_ha = nn.Sequential(
            nn.Linear(opt.ha_dim, opt.ha_dim*2),
            nn.BatchNorm1d(opt.ha_dim*2),
            nn.LeakyReLU(),

            nn.Linear(opt.ha_dim*2, opt.ha_dim*4),
            nn.BatchNorm1d(opt.ha_dim*4),
            nn.LeakyReLU(),

            nn.Linear(opt.ha_dim*4, self.out_dim),
            nn.BatchNorm1d(self.out_dim)
        )
        self.main_hg = nn.Sequential(
            nn.LeakyReLU(),

            nn.Linear(opt.hg_dim, opt.hg_dim*2),
            nn.BatchNorm1d(opt.hg_dim*2),
            nn.LeakyReLU(),

            nn.Linear(opt.hg_dim*2, opt.hg_dim*4),
            nn.BatchNorm1d(opt.hg_dim*4),
            nn.LeakyReLU(),

            nn.Linear(opt.hg_dim*4, opt.num_train),
            nn.BatchNorm1d(opt.num_train)
        )

    def forward(self, input,branch):
        if branch ==0:
            embedding = self.main_em(input).view(-1,self.out_dim)
        elif branch == 1:
            embedding = self.main_ha(input).view(-1, self.out_dim)
        elif branch == 2:
            embedding = self.main_hg(input).view(-1, opt.num_train)
        return embedding

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
    def __init__(self, hidden_dim=128, tagset_size=74):
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
gpu_num = int(input('Tell me the gpu you wanna use for this experiment:'))
parser.add_argument('--gpu', type=int, default=gpu_num)
parser.add_argument('--siter', type=int, default=0, help='number of itr to start with')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--data_root',
                     default='/media/tony/STORAGE/DATA/CASIAB-OLD-RGB-BS/',
                    # default='/media/tony/MyBook-MSU-CVLAB/CASIA/DatasetB/seg_rgb_mrcnn/',
                    # default='/user/zhang835/link2cvl-tony/CASIAB-OLD-RGB-BS',
                     help='root directory for data')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--em_dim', type=int, default=320, help='size of the pose vector')
parser.add_argument('--ha_dim', type=int, default=288, help='size of the appearance vector')
parser.add_argument('--hg_dim', type=int, default=32, help='size of the gait vector')
parser.add_argument('--image_width', type=int, default=64, help='the width of the input image to network')
parser.add_argument('--image_height', type=int, default=64, help='the height of the input image to network')
parser.add_argument('--max_step', type=int, default=20, help='maximum distance between frames')
parser.add_argument('--data_threads', type=int, default=1, help='number of parallel data loading threads')
parser.add_argument('--normalize', action='store_true', help='if true, normalize pose vector')
parser.add_argument('--num_train',type=int, default=74, help='')
import datetime
time_now = str(datetime.datetime.now())
parser.add_argument('--signature', default=time_now)
parser.add_argument('--savedir', default='../runs')
opt = parser.parse_args()
print(opt)
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
# os.environ["CUDA_LAUNCH_BLOCKING"] = str(opt.gpu)
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
clfer = classification()
netE.apply(init_weights)
netD.apply(init_weights)
lstm.apply(init_weights)
clfer.apply(init_weights)
if opt.siter is not 0:
    checkpoint = torch.load('%s/modules/%s/%d' % (opt.savedir,opt.signature, opt.siter))
    netE = checkpoint['netE']
    netD = checkpoint['netD']
    lstm = checkpoint['lstm']
    clfer = checkpoint['clfer']
    print('model loadinged successfully')
optimizerE = optim.Adam(netE.parameters(), lr=opt.lr, betas=(0.9, 0.999),weight_decay=0.001)
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.9, 0.999),weight_decay=0.001)
optimizerLstm = optim.Adam(lstm.parameters(), lr=opt.lr, betas=(0.9, 0.999),weight_decay=0.001)
optimizerClfer = optim.Adam(clfer.parameters(), lr=opt.lr, betas=(0.9, 0.999),weight_decay=0.001)
mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()
cse_loss = nn.CrossEntropyLoss()
trp_loss = nn.TripletMarginLoss(margin=2.0)
netE.cuda()
netD.cuda()
lstm.cuda()
clfer.cuda()
mse_loss.cuda()
bce_loss.cuda()
cse_loss.cuda()
trp_loss.cuda()

# l1_crit = nn.L1Loss(size_average=False)
# reg_loss = 0
# for param in netE.parameters():
#     reg_loss += l1_crit(param)
#
# factor = 0.0005
# loss = factor * reg_loss
# #################################################################################################################
# DATASET PREPARATION
def get_training_batch(data_loader):
    while True:
        for sequence in data_loader:
            batch = sequence[0].cuda(),sequence[1].cuda(),sequence[2].cuda(),sequence[3].cuda()
            yield batch


train_data1 = CASIAB(
    train=True,
    data_root=opt.data_root,
    sequence_len=opt.max_step,
    n_train=opt.num_train
)
train_loader1 = DataLoader(train_data1,
                          num_workers=opt.data_threads,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)

training_batch_generator1 = get_training_batch(train_loader1)

# train_data2 = CASIAB-NEW(
#     train=True,
#     data_root=opt.data_root,
#     n_train=24
# )
# train_loader2 = DataLoader(train_data2,
#                           num_workers=opt.data_threads,
#                           batch_size=opt.batch_size,
#                           shuffle=True,
#                           drop_last=True,
#                           pin_memory=True)
#
# training_batch_generator2 = get_training_batch(train_loader2)

test_data = CASIAB(
    train=False,
    data_root=opt.data_root,
    sequence_len=opt.max_step,
    n_train=opt.num_train
)
test_loader = DataLoader(test_data,
                         num_workers=opt.data_threads,
                         batch_size=opt.batch_size,
                         shuffle=True,
                         drop_last=True,
                         pin_memory=True)
testing_batch_generator = get_training_batch(test_loader)

#################################################################################################################

def make_analogy(x):
    # (B, L, C, H, W)
    # (   B, C, H, W)
    netE.eval()
    netD.eval()

    def rand_idx():
        return np.random.randint(0, opt.batch_size)
        # return 0

    def rand_step():
        return np.random.randint(0, opt.max_step)
        # return 0

    none = torch.zeros([1, 3, 64, 64]).cuda()
    x_gs = torch.stack([i for i in [x[0][rand_step()], x[0][rand_step()], x[0][rand_step()], x[0][rand_step()],
                                    x[rand_idx()][rand_step()], x[rand_idx()][rand_step()], x[rand_idx()][rand_step()],
                                    x[rand_idx()][rand_step()]]]).cuda()
    h_gs = netE(x_gs)[1]
    # h_gs = torch.zeros(8,20).cuda()

    x_as = torch.stack([x[i][rand_step()] for i in [0, rand_idx(), rand_idx(), rand_idx(), rand_idx()]]).cuda()
    # x_as = torch.stack([x[i][0] for i in [2, 2, 2, 2, 2]]).cuda()

    h_as = netE(x_as)[0]
    # h_as = torch.ones(5,128).cuda()


    gene = [netD(torch.stack([i] * 8).cuda(), h_gs) for i in h_as]
    row0 = torch.cat([none, x_gs])
    rows = [torch.cat([e.unsqueeze(0), gene[i]]) for i, e in enumerate(x_as)]
    to_plot = torch.cat([row0] + rows)


    img = make_grid(to_plot,9)
    return img

def plot_anology(train,test,epoch):
    train_anology = make_analogy(train)
    test_anology = make_analogy(test)
    all = torch.cat([train_anology,test_anology],dim=1)
    writer.add_image('Image', all, epoch)
    fname = '%s/analogy/%s/%d.png' % (opt.savedir,opt.signature,epoch)
    save_image(all, fname, 9)

def eval_lstm_cmc(glr, prb):
    hp_glr = [netE(glr[i].cuda())[1].detach() for i in range(len(glr))]
    hp_glr = torch.stack(hp_glr, 0).view(len(hp_glr), -1, opt.hg_dim)
    glr_vec = lstm(hp_glr)[1].detach().cpu().numpy()

    hp_prb = [netE(prb[i].cuda())[1].detach() for i in range(len(prb))]
    hp_prb = torch.stack(hp_prb, 0).view(len(hp_prb), -1, opt.hg_dim)
    prb_vec = lstm(hp_prb)[1].detach().cpu().numpy()

    score = []
    for i in range(len(prb_vec)):
        id = (i // (1*1)) +1
        id_range = list(range((id-1)*1*1,id*1*1))
        print(id_range)
        score.append(calculate_identication_rate_single(glr_vec,prb_vec[i],id_range)[0])

    return sum(score)/float(len(score))

def write_tfboard(vals,itr,name):
    for idx,item in enumerate(vals):
        writer.add_scalar('data/%s%d'%(name,idx), item, itr)

#################################################################################################################
# TRAINING FUNCTION DEFINE
def train_main(Xn,Xc,Xmx,l):
    l2 = l
    Xn,Xc,Xmx = Xn.transpose(0, 1),Xc.transpose(0, 1),Xmx.transpose(0, 1)
    accu = []
    haha = []
    self_rec_loss=0
    for i in range(5,len(Xn)):
        netE.zero_grad()
        netD.zero_grad()
        lstm.zero_grad()
        clfer.zero_grad()
        # rp = torch.randperm(opt.batch_size).cuda()
        rdm = torch.LongTensor(1).random_(5, len(Xn))[0]
        # ------------------------------------------------------------------
        xmx0, xmx1 = Xmx[rdm], Xmx[i]
        hamx0, hgmx0, emmx0 = netE(xmx0)
        hamx1, hgmx1, emmx1 = netE(xmx1)
        # ------------------------------------------------------------------
        # sim_loss = mse_loss(hamx0, hamx1.detach())
        # loss = sim_loss*0.1

        # xmx0_ = netD(hamx0, hgmx0)
        # self_rec_loss += mse_loss(xmx0_, xmx0)

        xmx1_ = netD(hamx0,hgmx1)
        self_rec_loss += mse_loss(xmx1_,xmx1)


        accu.append(hamx1)
        haha.append(hgmx1)

    accu = torch.stack(accu)
    accu = torch.mean(accu,0)
    out = clfer(accu,1)
    loss_out = cse_loss(out, l2) / 20

    haha = torch.stack(haha)
    haha = torch.mean(haha,0)
    out_haha = clfer(haha,2)
    loss_out_haha = cse_loss(out_haha, l) / 20

    # xmx1_ = netD(hamx0,hgmx1)
    # cross_rec_loss = mse_loss(xmx1_,xmx1)

    # ------------------------------------------------------------------
    # xn0, xn1, xc0, xc1 = Xn[rdm], Xn[i], Xc[rdm], Xc[i]
    # # han0, hgn0 = netE(xn0)
    # han1, hgn1, emn1 = netE(xn1)
    # # hac0, hgc0 = netE(xc0)
    # hac1, hgc1, emc1 = netE(xc1)
    # ------------------------------------------------------------------

    loss = loss_out+self_rec_loss+loss_out_haha
    loss.backward()
    optimizerE.step()
    optimizerD.step()
    optimizerClfer.step()

    # writer.add_image('xn1_', xn1_, itr)
    # writer.add_image('xc1_', xc1_, itr)
    return [loss_out.data.cpu().numpy(),
            loss_out_haha.data.cpu().numpy(),
            self_rec_loss.data.cpu().numpy()]

def train_main2(Xn,Xc,lb,itr):

    def find_idx(glrs, aprb):
        scores = []
        for i in glrs:
            scores.append(calculate_cosine_similarity(i, aprb))
        max_val = max(scores)
        max_idx = scores.index(max_val)
        return max_idx

    def determin_the_right(array):
        matrix = np.zeros([len(array), len(array)], np.int)
        new_arr = []
        for i, e in enumerate(array):
            matrix[i, :] = np.array(list(range(e - i, e + len(array) - i)))
        for i in range(len(matrix)):
            a = matrix[:, i]
            b = itemfreq(a)
            pos = np.argmax(b[:, 1])
            val = b[pos, 0]
            if 0 <= val <= 19:
                new_arr.append(val)
            else:
                new_arr.append(-1)
        return new_arr
    # idx_batch = []

    for i in range(len(Xn)-1): #for each subject
        netE.zero_grad()
        netD.zero_grad()
        #  .zero_grad()

        xn, xc,xcd = Xn[i],Xc[i],Xc[i+1]
        _, hgn = netE(xn)
        _, hgc = netE(xc)
        _, hgcd = netE(xcd)
        idxs = []
        idxsd = []
        loss = 0
        ims = []
        ims_ = []
        for ii,e in enumerate(hgn):
            idx = find_idx(hgc.data.cpu().numpy(),e.data.cpu().numpy())
            idxs.append(idx)
        idxs = determin_the_right(idxs)

        for j in range(1,len(idxs)):
            # rdm1 = torch.LongTensor(1).random_(0, len(idxs))[0]
            # rdm2 = torch.LongTensor(1).random_(0, len(idxs))[0]
            # while idxs[rdm1] < 0 and idxsd[rdm2] < 0:
            #     rdm1 = torch.LongTensor(1).random_(0, len(idxs))[0]
            #     rdm2 = torch.LongTensor(1).random_(0, len(idxs))[0]
            if idxs[j] >= 0:
                # sn1 = netD(han[rdm].unsqueeze(0),hgc[idxs[j]].unsqueeze(0))
                # sn2 = netD(hac[idxs[rdm]].unsqueeze(0),hgn[j].unsqueeze(0))
                # loss+=mse_loss(sn1,xn[j].unsqueeze(0))+mse_loss(sn2,xc[idxs[j]].unsqueeze(0))

                # loss+=0.01*mse_loss(hgn[j].unsqueeze(0),hgc[idxs[j]].unsqueeze(0))

                # loss+=trp_loss(hgn[j].unsqueeze(0),hgc[idxs[j]].unsqueeze(0),hgcd[idxsd[j]].unsqueeze(0))
                # loss+=0.01*(cse_loss(hgnfc[j].unsqueeze(0),lb[i].unsqueeze(0))+cse_loss(hgcfc[idxs[j]].unsqueeze(0),lb[i].unsqueeze(0)))


                im1 = xn[j-1]
                im2 = xc[idxs[j-1]]
                im1 = im1.unsqueeze(0)
                dsada = netE(im1)
                loss+=0.01*mse_loss(netD(),im2.unsqueeze(0))

                im = torch.cat([im1, im2], dim=2)
                ims.append(im)
                # im_ = torch.cat([sn1.squeeze(0), sn2.squeeze(0)], dim=2)
                # ims_.append(im_)

        if len(ims) != 0:
            loss.backward()
            optimizerE.step()
            optimizerD.step()
    if len(ims) != 0:
        ims = torch.cat(ims, dim=1)
        # ims_ = torch.cat(ims_, dim=1)
        # imss = torch.cat([ims, ims_], dim=2)
        writer.add_image('ims', ims, itr)
    try:
        return [loss.data.cpu().numpy()]
    except:
        return [loss]

def train_lstm(x_n,x_c,x_mx,l):
    x_n = x_n.transpose(0, 1)
    x_c = x_c.transpose(0, 1)
    x_mx = x_mx.transpose(0, 1)
    cse=0
    mse = 0
    trp = 0
    hgs_nm = []
    hgs_cl = []
    hgs_mx = []
    for i in range(0, len(x_n)):
        netE.zero_grad()
        netD.zero_grad()
        lstm.zero_grad()
        clfer.zero_grad()
        factor = (i/5)**2/10

        # rp = torch.randperm(opt.batch_size).cuda()

        xmx = x_mx[i]
        hgs_mx.append(netE(xmx)[1])
        lstm_out_mx = lstm(torch.stack(hgs_mx))[0]
        cse += cse_loss(lstm_out_mx,Variable(l))*factor


        xn = x_n[i]
        hgs_nm.append(netE(xn)[1])
        hgs_nm_ = torch.stack(hgs_nm)
        lstm_out_n = lstm(hgs_nm_)[1]

        xc = x_c[i]
        hgs_cl.append(netE(xc)[1])
        lstm_out_c = lstm(torch.stack(hgs_cl))[1]
        mse += mse_loss(lstm_out_n,lstm_out_c.detach())*factor

        # trp += 0.1*trp_loss(out_n,out_c,out_c[rp,:])

    cse /= opt.max_step
    cse *= 0.1
    mse /= opt.max_step
    mse *= 0.1

    los = cse
    los.backward()
    optimizerLstm.step()
    optimizerE.step()
    return [cse.data.cpu().numpy(),mse.data.cpu().numpy()]

#################################################################################################################
# FUN TRAINING TIME !
train_eval = test_data.get_eval_data(True)
test_eval = test_data.get_eval_data(False)
writer = SummaryWriter('%s/logs/%s'%(opt.savedir,opt.signature))
itr = opt.siter
while True:
    netE.train()
    netD.train()
    lstm.train()
    clfer.train()

    im_nm, im_cl,im_mx,lb = next(training_batch_generator1)
    print(lb)

    losses1 = train_main(im_nm, im_cl,im_mx,lb)
    write_tfboard(losses1,itr,name='EDLoss')
    #
    # losses3 = train_main2(im_nm, im_cl,lb,itr)
    # write_tfboard(losses3, itr, name='HAHALoss')

    losses3 = train_lstm(im_nm,im_cl,im_mx,lb)
    write_tfboard(losses3, itr, name='LstmLoss')

    print(itr)

    # ----------------EVAL()--------------------
    if itr % 5 == 0:
        # with torch.no_grad():
        netD.eval()
        netE.eval()
        lstm.eval()
        clfer.eval()
        scores_cmc_cl = eval_lstm_cmc(train_eval[0], train_eval[1])
        write_tfboard([scores_cmc_cl], itr, name='train_accu_rank1_cl')
        scores_cmc_cl = eval_lstm_cmc(test_eval[0], test_eval[1])
        write_tfboard([scores_cmc_cl], itr, name='test_accu_rank1_cl')

        im_nm_, im_cl_, im_mx_, _= next(testing_batch_generator)
        plot_anology(im_mx,im_mx_,itr)

        # ----------------SAVE MODEL--------------------
    # if itr % 200 == 0:
    #     torch.save({
    #         'netD': netD,
    #         'netE': netE,
    #         'lstm':lstm
    #         },
    #         '%s/modules/%s/%d'%(opt.savedir,opt.signature,itr))

    itr+=1


