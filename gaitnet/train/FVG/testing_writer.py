
import random
import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
import argparse
import torchvision

parser = argparse.ArgumentParser()
gpu_index = []
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--data_root',
                    default='/home/tony/Research/NEW-MRCNN/SEG/',
                    help='root directory for data')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--em_dim', type=int, default=320, help='size of the pose vector')
parser.add_argument('--fa_dim', type=int, default=288, help='size of the appearance vector')
parser.add_argument('--fg_dim', type=int, default=32, help='size of the gait vector')
parser.add_argument('--im_height', type=int, default=64, help='the height of the input image to network')
parser.add_argument('--im_width', type=int, default=32, help='the width of the input image to network')
parser.add_argument('--max_step', type=int, default=20, help='maximum distance between frames')
parser.add_argument('--savedir', default='./runs')
# signature = input('Specify a NAME for this running:')
# parser.add_argument('--signature', default=signature)
opt = parser.parse_args()

class FVG_instance(object):

    def __init__(self,
                 folder_path,
                 frame_height,
                 frame_width,
                 ):
        self.folder_path = folder_path
        self.frame_names = sorted(os.listdir(self.folder_path))
        self.frame_names = [f for f in self.frame_names if f.endswith('.png')]
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.frame_height, self.frame_width)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        try:
            frame = np.asarray(Image.open(os.path.join(self.folder_path,
                                                       self.frame_names[index])))
            frame = self.transform(frame)
        except:
            frame = torch.zeros(3, self.frame_height, self.frame_width)
        return frame
    def __len__(self):
        return len(self.frame_names)

class FVG(object):

    def __init__(self,
                 data_root,
                 clip_len,
                 im_height,
                 im_width,
                 seed,
                 ):

        np.random.seed(seed)
        self.is_train_data = True
        self.data_root = data_root
        self.clip_len = clip_len
        self.im_height = im_height
        self.im_width = im_width
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.im_height, self.im_width)),
            transforms.ToTensor()
        ])
        self.im_shape = [self.clip_len, 3, self.im_height, self.im_width]

        subjects = list(range(1, 226 + 1))
        random.Random(1).shuffle(subjects)
        self.train_subjects = subjects[:136]  # 136 subjects 60%
        self.test_subjects = subjects[136:]  # 90 subjects 40%
        print('training_subjects', self.train_subjects, len(self.train_subjects))
        print('testing_subjects', self.test_subjects, len(self.test_subjects))

    def get_eval_data(self,
                      is_train_data,
                      gallery,
                      probe_session1,
                      probe_session2,
                      ):
        if is_train_data is True:
            subjects = self.train_subjects
        else:
            subjects = self.test_subjects

        gt = []

        def read_frames(si_idx, vi_idx):
            if si_idx in list(range(1, 147 + 1)):
                reading_dir = 'session1'
            else:
                reading_dir = 'session2'

            in_path = os.path.join(self.data_root, reading_dir, '%03d_%02d' % (si_idx, vi_idx))
            fvg_instance = FVG_instance(folder_path=in_path,
                                        frame_height=self.im_height,
                                        frame_width=self.im_width)
            loader = DataLoader(fvg_instance,
                                num_workers=8,
                                batch_size=30,
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True)

            data = []
            for batch_frame in loader:
                batch_frame = batch_frame.cuda()
                data.append(batch_frame)
            if len(data):
                data = torch.cat(data)
            else:
                data = torch.zeros([3,self.im_height,self.im_width])
            return data

        test_data_glr = []
        test_data_prb = []

        for i, id in enumerate(subjects):
            test_data_glr.append(read_frames(id, gallery))
            if id in list(range(1, 147 + 1)):
                probes = probe_session1
            else:
                probes = probe_session2
            gt.append(len(probes))
            for j in probes:
                test_data_prb.append(read_frames(id, j))
            print('Training data?:', is_train_data, 'Reading', i, 'th subject:',id)

        test_data_glr = pad_sequence(test_data_glr)
        test_data_glr = test_data_glr[:70]
        test_data_prb = pad_sequence(test_data_prb)
        test_data_prb = test_data_prb[:70]
        return test_data_glr, \
               test_data_prb, \
               gt

    def __getitem__(self, index):
        """
        chose any two videos for the same subject
        :param index:
        :return:
        """
        training_data_structure = {
            'session1': list(range(1, 12 + 1)),
            'session2': list(range(1, 9 + 1)),
            'session3': list(range(1, 9 + 1)),
        }

        # 30 x 3 x 32 x 64
        def random_si_idx():
            if self.is_train_data:
                si_idx = np.random.choice(self.train_subjects)
                labels = self.train_subjects.index(si_idx)
            else:

                si_idx = np.random.choice(self.test_subjects)
                labels = self.test_subjects.index(si_idx)
            return si_idx, labels

        def random_vi_idx(si):

            if si in list(range(1, 147 + 1)):
                if si in [1, 2, 4, 7, 8, 12, 13, 17, 31, 40, 48, 77]:
                    reading_dir = random.choice(['session1', 'session3'])
                else:
                    reading_dir = 'session1'
            else:
                reading_dir = 'session2'

            vi_idx = np.random.choice(training_data_structure[reading_dir])

            return reading_dir, vi_idx

        def random_length(dirt, length):
            files = sorted(os.listdir(dirt))
            num = len(files)
            if num - length < 2:
                return None
            start = np.random.randint(1, num - length)
            end = start + length
            return files[start:end]

        def read_frames(frames_pth, file_names):
            # frames = np.zeros(self.im_shape, np.float32)
            frames = []
            for f in file_names:
                frame = np.asarray(Image.open(os.path.join(frames_pth, f)))
                frame = self.transform(frame)
                frames.append(frame)
            frames = torch.stack(frames)
            return frames

        si, labels = random_si_idx()
        session_dir1, vi1 = random_vi_idx(si)
        session_dir2, vi2 = random_vi_idx(si)
        frames_pth1 = os.path.join(self.data_root, session_dir1, '%03d_%02d' % (si, vi1))
        frames_pth2 = os.path.join(self.data_root, session_dir2, '%03d_%02d' % (si, vi2))
        file_names1 = random_length(frames_pth1, self.clip_len)
        file_names2 = random_length(frames_pth2, self.clip_len)

        while True:
            if file_names1 == None or file_names2 == None:
                session_dir1, vi1 = random_vi_idx(si)
                session_dir2, vi2 = random_vi_idx(si)
                frames_pth1 = os.path.join(self.data_root, session_dir1, '%03d_%02d' % (si, vi1))
                frames_pth2 = os.path.join(self.data_root, session_dir2, '%03d_%02d' % (si, vi2))
                file_names1 = random_length(frames_pth1, self.clip_len)
                file_names2 = random_length(frames_pth2, self.clip_len)
            else:
                break

        data1 = read_frames(frames_pth1, file_names1)
        data2 = read_frames(frames_pth2, file_names2)

        return data1, data2, labels

    def __len__(self):
        if self.is_train_data:
            return len(self.train_subjects) * 12
        else:
            return len(self.test_subjects) * 12



fvg = FVG(
    data_root=opt.data_root,
    clip_len=opt.max_step,
    im_height=opt.im_height,
    im_width=opt.im_width,
    seed=opt.seed
)
data_loader = DataLoader(fvg,
                       num_workers=8,
                       batch_size=opt.batch_size,
                       shuffle=True,
                       drop_last=True,
                       pin_memory=True)
def get_batch(is_train):
    fvg.is_train_data = is_train
    while True:
        for batch in data_loader:
            batch = [e.cuda() for e in batch]
            yield batch

training_batch_generator = get_batch(is_train=True)
testing_batch_generator = get_batch(is_train=False)

# ##SAVE
# proto_WS = fvg.get_eval_data(False, 2, probe_session1=list(range(4, 9 + 1)), probe_session2=list(range(4, 6 + 1)))
# # proto_CB = fvg.get_eval_data(False, 2, probe_session1=list(range(10, 12 + 1)), probe_session2=[])
# # proto_CL = fvg.get_eval_data(False, 2, probe_session1=[], probe_session2=list(range(7, 9 + 1)))
#
# proto_WS = proto_WS[0].cpu().numpy(), proto_WS[1].cpu().numpy(), proto_WS[2]
# np.save('testset_WS',proto_WS)
#
# # proto_CB = proto_CB[0].cpu().numpy(), proto_CB[1].cpu().numpy(), proto_CB[2]
# # np.save('testset_CB',proto_CB)

##READ

proto_WS = np.load('testset_WS.npy',allow_pickle=True)
proto_WS = torch.tensor(proto_WS[0]).cuda(), torch.tensor(proto_WS[1]).cuda(), proto_WS[2]
proto_CB = np.load('testset_CB.npy',allow_pickle=True)
proto_CB = torch.tensor(proto_CB[0]).cuda(), torch.tensor(proto_CB[1]).cuda(), proto_CB[2]
proto_CL = np.load('testset_CL.npy',allow_pickle=True)
proto_CL = torch.tensor(proto_CL[0]).cuda(), torch.tensor(proto_CL[1]).cuda(), proto_CL[2]
torchvision.utils.save_image(proto_CL[0][0], 'gallery.png',1)
torchvision.utils.save_image(proto_CL[1][0], 'probe.png',1)
# # pass



