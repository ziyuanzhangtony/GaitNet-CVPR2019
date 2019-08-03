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
from torchvision.utils import make_grid, save_image
import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter


class CASIAB(object):

    def __init__(self,
                 train,
                 data_root,
                 glr_views,
                 prb_views,
                 sequence_len,
                 image_width,
                 image_height,
                 n_train,
                 seed=1,
                 ):

        np.random.seed(seed)
        self.train = train
        self.data_root = data_root
        self.sequence_len = sequence_len
        self.image_width = image_width
        self.image_height = image_height
        self.n_train = n_train

        self.glr_views = glr_views
        self.prb_views = prb_views

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Pad((50, 0)),
            transforms.Resize((image_height,image_width)),
            transforms.ToTensor()
        ])

        subjects = list(range(1, 124 + 1))
        self.train_subjects = subjects[:n_train]
        self.test_subjects = subjects[n_train:]
        # self.test_subjects = subjects[:n_train]
        # self.train_subjects = subjects[:5]
        # self.test_subjects = subjects[:5]
        print('training_subjects', self.train_subjects, np.sum(self.train_subjects), len(self.train_subjects))
        print('testing_subjects', self.test_subjects, np.sum(self.test_subjects), len(self.test_subjects))

    def __getitem__(self, index):
        shape = [self.sequence_len, 3, self.image_height, self.image_width]

        def random_chose():
            if self.train:
                si_idx = np.random.choice(self.train_subjects)
                label = self.train_subjects.index(si_idx)
            else:
                si_idx = np.random.choice(self.test_subjects)
                label = self.test_subjects.index(si_idx)

            cond1 = np.random.choice([1, 2])
            if cond1 == 1:
                senum1 = np.random.choice(list(range(1, 2 + 1)))
            else:
                senum1 = np.random.choice(list(range(1, 2 + 1)))

            cond2 = np.random.choice([1, 2])
            if cond2 == 1:
                senum2 = np.random.choice(list(range(1, 2 + 1)))
            else:
                senum2 = np.random.choice(list(range(1, 2 + 1)))

            # view_idx1 = np.random.choice(list(range(0, 180 + 1, 18)))
            # view_idx2 = np.random.choice(list(range(0, 180 + 1, 18)))

            view_idx1 = np.random.choice([90])
            view_idx2 = np.random.choice([90])

            return si_idx,(cond1,senum1,view_idx1),(cond2,senum2,view_idx2),label

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

        si_idx, param1, param2, label = random_chose()
        video_in_nm = os.path.join(self.data_root,
                                   '%03d-%1d-%02d-%03d' % (si_idx,
                                                           param1[0],
                                                           param1[1],
                                                           param1[2]))
        start_nm, end_nm, files_nm = random_length(video_in_nm, self.sequence_len)

        video_in_cl = os.path.join(self.data_root,
                                   '%03d-%1d-%02d-%03d' % (si_idx,
                                                           param2[0],
                                                           param2[1],
                                                           param2[2]))
        start_cl, end_cl, files_cl = random_length(video_in_cl, self.sequence_len)

        while True:
            if start_nm == end_nm == 0 or start_cl == end_cl == 0:
                si_idx, param1, param2, label = random_chose()
                video_in_nm = os.path.join(self.data_root,
                                           '%03d-%1d-%02d-%03d' % (si_idx,
                                                                   param1[0],
                                                                   param1[1],
                                                                   param1[2]))
                start_nm, end_nm, files_nm = random_length(video_in_nm, self.sequence_len)

                video_in_cl = os.path.join(self.data_root,
                                           '%03d-%1d-%02d-%03d' % (si_idx,
                                                                   param2[0],
                                                                   param2[1],
                                                                   param2[2]))
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
        return imgs_nm, imgs_cl,imgs_mx, label

    def get_eval_data(self,train=False):
        test_data_glr_list = []
        test_data_prb_list = []

        def read_frames_(si, ci, sei, vi):
            video_in = os.path.join(self.data_root, '%03d-%1d-%02d-%03d' % (si, ci, sei, vi))
            files = sorted(os.listdir(video_in))
            shape = [len(files), 3, self.image_height, self.image_width]
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

        conditions = [1,2]
        ###########################################################
        #gallry
        for vi in self.glr_views:
            test_data_glr_this = []
            for i, id in enumerate(data_type):
                test_data_glr_this.append(read_frames_(id, conditions[0], 1, vi))
                print(vi, i)
            test_data_glr_this = pad_sequences(test_data_glr_this, maxlen=70, dtype='float32', padding='post')
            test_data_glr_list.append(test_data_glr_this)


        ###########################################################
        # probe
        for vi in self.prb_views:
            test_data_prb_this = []
            for i, id in enumerate(data_type):
                print(vi, i)
                test_data_prb_this.append(read_frames_(id, conditions[1], 1, vi))
            test_data_prb_this = pad_sequences(test_data_prb_this, maxlen=70, dtype='float32', padding='post')
            test_data_prb_list.append(test_data_prb_this)

        return torch.tensor(test_data_glr_list).permute(0, 2, 1, 3, 4, 5), \
               torch.tensor(test_data_prb_list).permute(0, 2, 1, 3, 4, 5)

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


def process_confusion_matrix(matrix, n_class, gt):
    matrix = np.reshape(matrix, (n_class * sum(gt)))

    def make_labels():
        squre_matrix = np.eye(n_class, n_class)
        matrix = []

        for i in range(n_class):
            for j in range(gt[i]):
                matrix.append(squre_matrix[i])

        matrix = np.asarray(matrix)
        return np.concatenate(matrix)

    labels = make_labels()
    labels = np.reshape(labels, (n_class * sum(gt)))
    fpr, tpr, _ = roc_curve(labels, matrix)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def plot_roc(fpr, tpr, roc_auc):
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


def find_idx(fpr, tpr, threthold=[0.01, 0.05, 0.1], ifround=True):
    outptut = []
    for i in threthold:
        item = fpr[fpr < i + 0.005].max()
        idx = np.where(fpr == item)
        val = tpr[idx][-1]
        if ifround:
            val = round(val, 2)
        outptut.append(val)
    return outptut


def calculate_cosine_similarity(a, b):
    score = 1 - spatial.distance.cosine(a, b)
    return score


def calculate_identication_rate_single(glrs, aprb, trueid, rank=1):
    scores = []
    for i in glrs:
        scores.append(calculate_cosine_similarity(i, aprb))
    max_val = max(scores)
    max_idx = scores.index(max_val)

    right, predicted = trueid, max_idx
    print(right, predicted)

    if max_idx in trueid:
        return 1, [right, predicted]
    else:
        return 0, [right, predicted]


def adjust_white_balance(x):
    x = x.cpu()
    avgR = np.average(x[0, :, :])
    avgG = np.average(x[1, :, :])
    avgB = np.average(x[2, :, :])

    avg = (avgB + avgG + avgR) / 3

    x[0, :, :] = np.minimum(x[0] * (avg / avgR), 1)
    x[1, :, :] = np.minimum(x[1] * (avg / avgG), 1)
    x[2, :, :] = np.minimum(x[2] * (avg / avgB), 1)

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
            nn.Conv2d(nin, nout, 4, 2, 1
                      ),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2),
        )

    def forward(self, input):
        return self.main(input)


class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_upconv, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nin, nout, 4, 2, 1, ),
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
            vgg_layer(nf, nf),

            dcgan_conv(nf, nf * 2),
            vgg_layer(nf * 2, nf * 2),

            dcgan_conv(nf * 2, nf * 4),
            vgg_layer(nf * 4, nf * 4),

            dcgan_conv(nf * 4, nf * 8),
            vgg_layer(nf * 8, nf * 8),

            # nn.Conv1d(nf * 8, self.em_dim, 4, 1, 0),
            # nn.BatchNorm2d(self.em_dim),
        )

        self.flatten = nn.Sequential(
            nn.Linear(nf * 8 * 2 * 4,self.em_dim),
            nn.BatchNorm1d(self.em_dim),
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
            nn.Linear(self.em_dim, self.em_dim // 2),
            nn.BatchNorm1d(self.em_dim // 2),

            nn.LeakyReLU(),
            nn.Linear(self.em_dim // 2, self.em_dim // 2),
            nn.BatchNorm1d(self.em_dim // 2),

            nn.LeakyReLU(),
            nn.Linear(self.em_dim // 2, opt.hg_dim),
            nn.BatchNorm1d(opt.hg_dim)
        )

    def forward(self, input):
        embedding = self.main(input).view(-1, 64 * 8 * 2 * 4)
        embedding = self.flatten(embedding)
        ha, hg = torch.split(embedding, [opt.ha_dim, opt.hg_dim], dim=1)
        return ha, hg, embedding


class decoder(nn.Module):
    def __init__(self, nc=3):
        super(decoder, self).__init__()
        nf = 64
        self.em_dim = opt.em_dim

        self.trans = nn.Sequential(
            nn.Linear(self.em_dim, nf * 8 * 2 * 4),
            nn.BatchNorm1d(nf * 8 * 2 * 4),
        )

        self.main = nn.Sequential(
            # nn.ConvTranspose2d(self.em_dim, nf * 8, 4, 1, 0),
            # nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2),
            vgg_layer(nf * 8, nf * 8),

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



    def forward(self, ha, hg):
        hidden = torch.cat([ha, hg], 1).view(-1, opt.em_dim)
        small = self.trans(hidden).view(-1, 64 * 8, 4, 2)
        img = self.main(small)
        return img


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
        lstm_out, _ = self.lstm(batch.view(lens, -1, self.source_dim))
        lstm_out_test = self.fc1(torch.mean(lstm_out.view(lens, -1, self.hidden_dim), 0))
        # lstm_out_test = self.fc1(lstm_out.view(lens,-1,self.hidden_dim)[-1])
        # lstm_out_test = torch.mean(batch.view(lens,-1,64),0)
        lstm_out_train = self.main(lstm_out_test).view(-1, self.tagset_size)
        return lstm_out_train, lstm_out_test, lstm_out


#################################################################################################################
# HYPER PARAMETERS INITIALIZING
parser = argparse.ArgumentParser()
gpu_num = int(input('Enter the gpu for this experiment:'))
parser.add_argument('--gpu', type=int, default=gpu_num)
parser.add_argument('--siter', type=int, default=24000, help='number of itr to start with')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--data_root',
                    default='/home/tony/Research/CASIAB-OLD-RGB-BS',
                    help='root directory for data')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--em_dim', type=int, default=320, help='size of the pose vector')
parser.add_argument('--ha_dim', type=int, default=288, help='size of the appearance vector')
parser.add_argument('--hg_dim', type=int, default=32, help='size of the gait vector')
parser.add_argument('--image_width', type=int, default=32, help='the width of the input image to network')
parser.add_argument('--image_height', type=int, default=64, help='the height of the input image to network')
parser.add_argument('--max_step', type=int, default=50, help='maximum distance between frames')
parser.add_argument('--data_threads', type=int, default=8, help='number of parallel data loading threads')
# parser.add_argument('--normalize', action='store_true', help='if true, normalize pose vector')
parser.add_argument('--num_train',type=int, default=74, help='')
parser.add_argument('--glr_views',type=list, default=list(range(0, 180 + 1, 18)), help='')
parser.add_argument('--prb_views',type=list, default=[0,54,90,126], help='')


time_now = "2019-04-11 12:52:33.752633"
parser.add_argument('--signature', default=time_now)
parser.add_argument('--savedir', default='./runs')
opt = parser.parse_args()
print(opt)
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
os.makedirs('%s/analogy/%s' % (opt.savedir, opt.signature), exist_ok=True)
os.makedirs('%s/modules/%s' % (opt.savedir, opt.signature), exist_ok=True)
#################################################################################################################
# MODEL PROCESS
netE = encoder()
netD = decoder()
lstm = lstm()
netE.apply(init_weights)
netD.apply(init_weights)
lstm.apply(init_weights)
if opt.siter is not 0:
    checkpoint = torch.load('%s/modules/%s/%d.pickle' % (opt.savedir, opt.signature, opt.siter))
    netE.load_state_dict(checkpoint['netE'])
    netD.load_state_dict(checkpoint['netD'])
    lstm.load_state_dict(checkpoint['lstm'])
    print('model loadinged successfully')

optimizerE = optim.Adam(netE.parameters(), lr=opt.lr, betas=(0.9, 0.999),weight_decay=0.001)
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.9, 0.999),weight_decay=0.001)
optimizerLstm = optim.Adam(lstm.parameters(), lr=opt.lr, betas=(0.9, 0.999),weight_decay=0.001)

# optimizerE = optim.Adam(netE.parameters(), lr=opt.lr, betas=(0.9, 0.999))
# optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.9, 0.999))
# optimizerLstm = optim.Adam(lstm.parameters(), lr=opt.lr, betas=(0.9, 0.999))

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
            batch = sequence[0].cuda(),sequence[1].cuda(),sequence[2].cuda(),sequence[3].cuda()
            yield batch


train_data1 = CASIAB(
    train=True,
    data_root=opt.data_root,
    sequence_len=opt.max_step,
    glr_views=opt.glr_views,
    prb_views=opt.prb_views,
    image_height=opt.image_height,
    image_width=opt.image_width,
    n_train=opt.num_train
)
train_loader1 = DataLoader(train_data1,
                          num_workers=opt.data_threads,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)

training_batch_generator1 = get_training_batch(train_loader1)

test_data = CASIAB(
    train=False,
    data_root=opt.data_root,
    sequence_len=opt.max_step,
    glr_views=opt.glr_views,
    prb_views=opt.prb_views,
    image_height=opt.image_height,
    image_width=opt.image_width,
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

def make_anamition(x):

    for i in range(x.shape[1]):
        frame = make_analogy_inter_subs(x,i)
        frame = adjust_white_balance(frame.detach())
        save_image(frame,'%s/analogy/%s/%03d.png' % (opt.savedir, opt.signature,i))

def make_analogy_inter_subs(x,step):
    # (B, L, C, H, W)
    # (   B, C, H, W)

    def rand_step():
        return step

    none = torch.zeros([1, 3, 64, 32]).cuda()
    x_gs = torch.stack([i for i in [x[0][rand_step()], x[1][rand_step()], x[2][rand_step()], x[3][rand_step()]]]).cuda()
    h_gs = netE(x_gs)[1]
    # h_gs = torch.zeros(4,32).cuda()

    x_as = torch.stack([x[i][rand_step()] for i in [3, 4, 5]]).cuda()
    # x_as = torch.stack([x[i][0] for i in [2, 2, 2, 2, 2]]).cuda()

    h_as = netE(x_as)[0]
    # h_as = torch.zeros(3,288).cuda()

    gene = [netD(torch.stack([i] * 4).cuda(), h_gs) for i in h_as]
    row0 = torch.cat([none, x_gs])
    rows = [torch.cat([e.unsqueeze(0), gene[i]]) for i, e in enumerate(x_as)]
    to_plot = torch.cat([row0] + rows)

    img = make_grid(to_plot, 5)
    return img



#################################################################################################################
# FUN TRAINING TIME !
# train_eval = test_data.get_eval_data(True)
netD.eval()
netE.eval()
lstm.eval()
im_nm_, im_cl_, im_mx_, _= next(testing_batch_generator)
make_anamition(im_mx_)