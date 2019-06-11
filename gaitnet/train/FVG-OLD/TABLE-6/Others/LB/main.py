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
import torch.nn.init as init
import datetime
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
#################################################################################################################

# def loadImage(path):
#     inImage = cv2.imread(path, 0)
#     info = np.iinfo(inImage.dtype)
#     inImage = inImage.astype(np.float) / info.max
#     iw = inImage.shape[1]
#     ih = inImage.shape[0]
#     if iw < ih:
#         inImage = cv2.resize(inImage, (126, int(126 * ih/iw)))
#     else:
#         inImage = cv2.resize(inImage, (int(126 * iw / ih), 126))
#     inImage = inImage[0:126, 0:126]
#     return torch.from_numpy(2 * inImage - 1).unsqueeze(0)

class FVG(object):

    def __init__(self,
                 train,
                 data_root,
                 video_i=range(1, 12 + 1),
                 image_width=126,
                 image_height=126,
                 seed=1,
                 ):

        np.random.seed(seed)
        self.data_root = data_root

        self.video_i = video_i

        self.image_width = image_width
        self.image_height = image_height

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad((50, 0)),
            transforms.Resize((image_width, image_width)),
            transforms.ToTensor()
        ])

        subjects = list(range(1, 226 + 1))
        random.Random(1).shuffle(subjects)
        self.train_subjects = subjects[:136]  # 136 subjects 60%
        self.test_subjects = subjects[136:]  # 90 subjects 40%
        print('training_subjects', self.train_subjects, np.sum(self.test_subjects))
        print('testing_subjects', self.test_subjects, np.sum(self.test_subjects))

        self.train = train

    def loadImage(self, path):
        img = imread(path)
        img = self.transform(img)
        return img

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

        def format_path( si_idx, vi_idx):
            if si_idx in list(range(1, 147 + 1)):
                reading_dir = 'GEI-S1'
            else:
                reading_dir = 'GEI-S2'

            video_in = os.path.join(self.data_root, reading_dir, '%03d_%02d.png' % (si_idx, vi_idx))
            return video_in

        test_data_glr = []
        test_data_prb = []

        for i, id in enumerate(subjects):
            img = self.loadImage(format_path(id, gallery))
            test_data_glr.append(img)
            if id in list(range(1, 147 + 1)):
                probes = session1_probe
            else:
                probes = session2_probe
            gt.append(len(probes))
            for j in probes:
                img = self.loadImage(format_path(id, j))
                test_data_prb.append(img)
            print('Training data:', train, '.Reading', i, 'th subject. ')
        
        # test_data_glr = pad_sequences(test_data_glr, maxlen=70, dtype='float32', padding='post')
        # test_data_prb = pad_sequences(test_data_prb, maxlen=70, dtype='float32', padding='post')
        test_data_glr = torch.stack(test_data_glr).permute(1, 0, 2, 3)
        test_data_prb = torch.stack(test_data_prb).permute(1, 0, 2, 3)
        return test_data_glr, \
               test_data_prb, \
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

        # def read_img(reading_dir, si_idx, vi_idx):
        #     if reading_dir == '':
        #         if si_idx in list(range(1, 147 + 1)):
        #             reading_dir = 'GEI-S1'
        #         else:
        #             reading_dir = 'GEI-S2'
        #
        #     video_in = os.path.join(self.data_root, reading_dir, '%03d_%02d' % (si_idx, vi_idx))
        #
        #     img = imread(video_in)
        #     img = self.transform(img)
        #
        #     # print(reading_dir, si_idx, vi_idx)
        #     return torch.from_numpy(img)

        def format_path(reading_dir, si_idx, vi_idx):
            if reading_dir == '':
                if si_idx in list(range(1, 147 + 1)):
                    reading_dir = 'GEI-S1'
                else:
                    reading_dir = 'GEI-S2'

            video_in = os.path.join(self.data_root, reading_dir, '%03d_%02d.png' % (si_idx, vi_idx))
            return video_in

        test_data_glr = []
        test_data_prb = []

        # subjects = subjects[:10]

        for i, id in enumerate(subjects):


            if id in list(range(1, 147 + 1)):
                img = self.loadImage(format_path('GEI-S1', id, gallery))
                test_data_glr.append(img)

                if id in [1, 2, 4, 7, 8, 12, 13, 17, 31, 40, 48, 77]: # if subject in both s1 and 3
                    for j in session1_probe:
                        img = self.loadImage(format_path('GEI-S1', id, j))
                        test_data_prb.append(img)
                    for j in session3_probe:
                        img = self.loadImage(format_path('GEI-S3', id, j))
                        test_data_prb.append(img)
                    gt.append(len(session1_probe) + len(session3_probe))
                    print(i, id, 'GEI-S1&3', len(session1_probe) + len(session3_probe))
                else: # if subject only in s1
                    for j in session1_probe:
                        img = self.loadImage(format_path('GEI-S1', id, j))
                        test_data_prb.append(img)
                    gt.append(len(session1_probe))
                    print(i, id, 'GEI-S1', len(session1_probe))
            else: # if subject only in s2
                img = self.loadImage(format_path('GEI-S2', id, gallery))
                test_data_glr.append(img)
                for j in session1_probe:
                    img = self.loadImage(format_path('GEI-S2', id, j))
                    test_data_prb.append(img)
                gt.append(len(session2_probe))
                print(i, id, 'GEI-S2', len(session2_probe))

        # test_data_glr = pad_sequences(test_data_glr,maxlen=70,dtype='float32',padding='post')
        # test_data_prb = pad_sequences(test_data_prb,maxlen=70,dtype='float32',padding='post')
        test_data_glr = torch.stack(test_data_glr).permute(1, 0, 2, 3)
        test_data_prb = torch.stack(test_data_prb).permute(1, 0, 2, 3)
        return test_data_glr, \
               test_data_prb, \
               gt

    def __getitem__(self, index):


        def random_si_idx():
            if self.train:
                si_idx = np.random.choice(self.train_subjects)
                labels = self.train_subjects.index(si_idx)
            else:
                si_idx = np.random.choice(self.test_subjects)
                labels = self.test_subjects.index(si_idx)
            return si_idx, labels

        def random_vi_idx(si):

            if si in list(range(1, 147 + 1)):
                if si in [1, 2, 4, 7, 8, 12, 13, 17, 31, 40, 48, 77]:
                    reading_dir = random.choice(['GEI-S1', 'GEI-S3'])
                else:
                    reading_dir = 'GEI-S1'
            else:
                reading_dir = 'GEI-S2'

            vi_idx = np.random.choice(self.video_i)

            return reading_dir, vi_idx

        is_same = bool(random.getrandbits(1))
        if is_same:
            label = torch.tensor(1.0)
            si1, _ = random_si_idx()
            si2 = si1
        else:
            label = torch.tensor(0.0)
            si1, _ = random_si_idx()
            si2, _ = random_si_idx()
        session_dir1, vi1 = random_vi_idx(si1)
        session_dir2, vi2 = random_vi_idx(si2)
        video_pth1 = os.path.join(self.data_root, session_dir1, '%03d_%02d.png' % (si1, 2))
        video_pth2 = os.path.join(self.data_root, session_dir2, '%03d_%02d.png' % (si2, vi2))

        imgs1 = self.loadImage(os.path.join(video_pth1))
        imgs2 = self.loadImage(os.path.join(video_pth2))

        return imgs1, imgs2, label

    def __len__(self):
        if self.train:
            return len(self.train_subjects) * 12
        else:
            return len(self.test_subjects) * 12

class LBNet_1(nn.Module):
    def __init__(self, nc=1):
        super(LBNet_1, self).__init__()
        self.W1 = nn.Sequential(
            nn.Conv2d(nc, 16, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.LocalResponseNorm(5, 0.0001, 0.75, 2),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.W2 = nn.Sequential(
            nn.Conv2d(nc, 16, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.LocalResponseNorm(5, 0.0001, 0.75, 2),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.convolutions = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.LocalResponseNorm(5, 0.0001, 0.75, 2),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 256, kernel_size=7, stride=1)
        )
        self.mlp = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(21 * 21 * 256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = x[:, 0, :, :].unsqueeze(1)
        x2 = x[:, 1, :, :].unsqueeze(1)
        x1 = self.W1(x1)
        x2 = self.W2(x2)
        x = self.convolutions(x1 + x2)
        x = x.view(-1, 21*21*256)
        x = self.mlp(x)
        return x

class LBNet(nn.Module):
    def __init__(self, nc=2):
        super(LBNet, self).__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(nc, 16, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.LocalResponseNorm(5, 0.0001, 0.75, 2),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 64, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.LocalResponseNorm(5, 0.0001, 0.75, 2),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 256, kernel_size=7, stride=1)
        )
        self.mlp = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(21 * 21 * 256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.convolutions(x)
        x = x.view(-1, 21 * 21 * 256)
        x = self.mlp(x)
        return x

#################################################################################################################
parser = argparse.ArgumentParser()
gpu_num = int(input('Enter the gpu for this experiment:'))
parser.add_argument('--gpu', type=int, default=gpu_num)
parser.add_argument('--siter', type=int, default=70000, help='number of itr to start with')
parser.add_argument('--lr', default=0.00001, type=float, help='learning rate')
parser.add_argument('--data_root',
                    default='/home/tony/Research/GEI-DATA',
                    help='root directory for data')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
# time_now = str(datetime.datetime.now())
time_now = "2019-04-02 03:03:44.567003"

parser.add_argument('--signature', default=time_now)
parser.add_argument('--savedir', default='./runs')
opt = parser.parse_args()

print(opt)
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
os.makedirs('%s/modules/%s' % (opt.savedir, opt.signature), exist_ok=True)

#################################################################################################################
lbnet = LBNet()
for mod in list(lbnet.children())[0].children():
    if isinstance(mod, nn.Conv2d):
        init.normal_(mod.weight, 0.0, 0.01)
        init.constant_(mod.bias, 0.0)

optimizer = optim.Adam(lbnet.parameters(), lr=opt.lr)

lbnet.cuda()

checkpoint = torch.load('%s/modules/%s/%d.pickle' % (opt.savedir, opt.signature, opt.siter))
lbnet.load_state_dict(checkpoint['lbnet'])
print('model loadinged successfully')

#################################################################################################################

def get_training_batch(data_loader):
    while True:
        for sequence in data_loader:
            batch = sequence[0].cuda(), sequence[1].cuda(), sequence[2].cuda()
            yield batch


train_data1 = FVG(
    train=True,
    data_root=opt.data_root
)

train_loader1 = DataLoader(train_data1,
                           num_workers=8,
                           batch_size=opt.batch_size,
                           shuffle=True,
                           drop_last=True,
                           pin_memory=True)

training_batch_generator1 = get_training_batch(train_loader1)
test_data = FVG(
    train=False,
    data_root=opt.data_root,
)
test_loader = DataLoader(test_data,
                         num_workers=8,
                         batch_size=opt.batch_size,
                         shuffle=True,
                         drop_last=True,
                         pin_memory=True)
testing_batch_generator = get_training_batch(test_loader)

#################################################################################################################
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

def eval_lstm_roc(glr, prb, gt, n_test=90):
    cat_imgs = np.zeros((prb.shape[1]*n_test,2,126,126), dtype=np.float32)
    cnt=0

    for j in range(prb.shape[1]):
        for i in range(n_test):
            img = torch.cat((glr[:, i, :, :], prb[:, j, :, :]), 0)
            cat_imgs[cnt] = img
            cnt+=1

    with torch.no_grad():
        scores = []
        sub_length = int(len(cat_imgs)/108)
        starting = 0
        for i in range(108):
            ending = starting+sub_length
            print(starting, ending)
            sub_imgs = torch.tensor(cat_imgs[starting:ending]).cuda()
            sub_scores = lbnet(sub_imgs)[:,0]
            scores.extend(sub_scores.data.cpu().numpy())
            starting+=sub_length


        # squre_matrix = np.eye(n_test, n_test)
        # matrix = []
        #
        # for i in range(n_test):
        #     for j in range(gt[i]):
        #         matrix.append(squre_matrix[i])
        #
        # matrix = np.asarray(matrix)
        # matrix = np.concatenate(matrix)


        obj_arr = np.zeros((prb.shape[1], n_test), dtype=np.float32)
        for i in range(n_test):
            for j in range(prb.shape[1]):
                obj_arr[j, i] = scores[j*n_test+i]
        # plt.imshow(obj_arr)
        # plt.show()
        fpr, tpr, roc_auc = process_confusion_matrix(obj_arr, n_test, gt)

        return find_idx(fpr, tpr)

def write_tfboard(vals, itr, name):
    for idx, item in enumerate(vals):
        writer.add_scalar('data/%s%d' % (name, idx), item, itr)


#################################################################################################################


def train_main(img1, img2, label):
    # Xn, Xc, Xmx = Xn.transpose(0, 1), Xc.transpose(0, 1), Xmx.transpose(0, 1)
    label = label.unsqueeze(1)
    img = torch.cat((img1, img2), 1)
    output = lbnet(img)
    loss = F.binary_cross_entropy(output, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return [loss.data.cpu().numpy()]

#################################################################################################################
# FUN TRAINING TIME !
# train_eval = test_data.get_eval_data(True)
# proto1 = test_data.get_eval_data(False, 2, session1_probe=list(range(4, 9 + 1)), session2_probe=list(range(4, 6 + 1)))
# proto2 = test_data.get_eval_data(False, 2, session1_probe=list(range(10, 12 + 1)))
# proto3 = test_data.get_eval_data(False, 2, session2_probe=list(range(7, 9 + 1)))
# proto4 = test_data.get_eval_data(False, 2, session2_probe=list(range(10, 12 + 1)))
proto5 = test_data.get_eval_data_all(False,2,cross_session=True)

# proto6 = test_data.get_eval_data(True, 2, session1_probe=list(range(4, 9 + 1)), session2_probe=list(range(4, 6 + 1)))

writer = SummaryWriter('%s/logs/%s' % (opt.savedir, opt.signature))
itr = opt.siter
while True:
    lbnet.train()
    im_cond1, im_cond2, lb = next(training_batch_generator1)
    losses1 = train_main(im_cond1, im_cond2, lb)
    write_tfboard(losses1, itr, name='Loss')

    # ----------------EVAL()--------------------
    if itr % 1000 == 0:
        lbnet.eval()

        # scores1 = eval_lstm_roc(proto1[0], proto1[1], proto1[2])
        # scores2 = eval_lstm_roc(proto2[0], proto2[1], proto2[2])
        # scores3 = eval_lstm_roc(proto3[0], proto3[1], proto3[2])
        # scores4 = eval_lstm_roc(proto4[0], proto4[1], proto4[2])
        scores5 = eval_lstm_roc(proto5[0], proto5[1], proto5[2])
        # scores6 = eval_lstm_roc(proto6[0], proto6[1], proto6[2],136)

        # write_tfboard(scores1[:2], itr, name='WS')
        # write_tfboard(scores2[:2], itr, name='CB-OLD')
        # write_tfboard(scores3[:2], itr, name='CL')
        # write_tfboard(scores4[:2], itr, name='CBG')
        write_tfboard(scores5[:2], itr, name='ALL')

        # write_tfboard(scores6[:2], itr, name='ONTRAIN')

        # im_cond1_, im_cond2_, _ = next(testing_batch_generator)

        # ----------------SAVE MODEL--------------------
    if itr % 10000 == 0:
        torch.save({
            'lbnet': lbnet.state_dict()
        },
            '%s/modules/%s/%d.pickle' % (opt.savedir, opt.signature, itr), )

    itr += 1