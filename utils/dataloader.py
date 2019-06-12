import os
import random

import numpy as np
import torch
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from PIL import Image

def imread(path):
    frame = np.asarray(Image.open(path))
    return frame

def get_training_batch(data_loader):
    while True:
        for sequences in data_loader:
            batch = [sequence.cuda() for sequence in sequences]
            yield batch

class CASIAB(object):

    def __init__(self, is_train_data, train_structure, test_structure, opt):
        self.is_train_data = is_train_data

        self.data_root = opt.data_root
        self.clip_len = opt.clip_len
        self.im_height = opt.im_height
        self.im_width = opt.im_width

        self.train_structure = train_structure
        self.test_structure = test_structure

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.im_height, self.im_width)),
            transforms.Pad((2, 4)),
            transforms.Resize((self.im_height, self.im_width)),
            transforms.ToTensor(),
        ])

        subjects = list(range(1, 124 + 1)) # 124 subjects
        self.train_subjects = subjects[:opt.num_train]
        self.test_subjects = subjects[opt.num_train:]

        print('Number of Training_Subjects', self.train_subjects, len(self.train_subjects))
        print('testing_subjects', self.test_subjects, len(self.test_subjects))

    def __getitem__(self, index): #new

        video_shape = [self.clip_len, 3, self.im_height, self.im_width]
        # conditions = self.train_structure['clip1'][1].union(self.train_structure['clip2'][1])


        # 30 x 3 x 32 x 64
        def random_chose():
            if self.is_train_data:
                si_idx = np.random.choice(self.train_subjects)
                label = self.train_subjects.index(si_idx)
            else:

                si_idx = np.random.choice(self.test_subjects)
                label = self.test_subjects.index(si_idx)

            view_idx1 = np.random.choice(self.train_structure['clip1'][0])
            cond1 = np.random.choice(self.train_structure['clip1'][1])
            senum1 = np.random.choice(self.train_structure['clip1'][2])

            view_idx2 = np.random.choice(self.train_structure['clip2'][0])
            cond2 = np.random.choice(self.train_structure['clip2'][1])
            senum2 = np.random.choice(self.train_structure['clip2'][2])

            return si_idx, (cond1, senum1, view_idx1), (cond2, senum2, view_idx2), label

        def random_length(dirt, length):
            files = sorted(os.listdir(dirt))
            num = len(files)
            if num - length < 2:
                return None
            start = np.random.randint(1, num - length)
            end = start + length
            return files[start:end]

        def read_video(frames_pth, file_names):
            # frames = np.zeros(self.im_shape, np.float32)
            frames = []
            for f in file_names:
                frame = np.asarray(Image.open(os.path.join(frames_pth, f)))
                frame = self.transform(frame)
                frames.append(frame)
            frames = torch.stack(frames)
            return frames

        si, param1, param2, label = random_chose()
        frames_pth1 = os.path.join(self.data_root,
                                   '%03d-%s-%02d-%03d' % (si,
                                                          param1[0],
                                                          param1[1],
                                                          param1[2]))
        frames_pth2 = os.path.join(self.data_root,
                                   '%03d-%s-%02d-%03d' % (si,
                                                          param2[0],
                                                          param2[1],
                                                          param2[2]))
        file_names1 = random_length(frames_pth1, self.clip_len)
        file_names2 = random_length(frames_pth2, self.clip_len)

        while True:
            if file_names1 == None or file_names2 == None:
                si, param1, param2, label = random_chose()
                frames_pth1 = os.path.join(self.data_root,
                                           '%03d-%s-%02d-%03d' % (si,
                                                                  param1[0],
                                                                  param1[1],
                                                                  param1[2]))
                frames_pth2 = os.path.join(self.data_root,
                                           '%03d-%s-%02d-%03d' % (si,
                                                                  param2[0],
                                                                  param2[1],
                                                                  param2[2]))
                file_names1 = random_length(frames_pth1, self.clip_len)
                file_names2 = random_length(frames_pth2, self.clip_len)
            else:
                break

        data1 = read_video(frames_pth1, file_names1)
        data2 = read_video(frames_pth2, file_names2)

        if np.random.choice([True, False]):
            data3 = data1.clone()
        else:
            data3 = data2.clone()

        return data1, data2, data3, label

    def get_eval_format(self):
        print('Loading evaluation format')

        print(self.is_train_data)
        if self.is_train_data:
            subjects = self.train_subjects
        else:
            subjects = self.test_subjects

        test_data_glr = []
        test_data_prb = []

        def read_video(si, con, sei, vi):
            folder_path = os.path.join(self.data_root, '%03d-%s-%02d-%03d' % (si, con, sei, vi))
            frame_names = sorted(os.listdir(folder_path))
            # random.shuffle(frame_names)
            # print(frame_names)
            frame_names = [f for f in frame_names if f.endswith('.png')]
            video_shape = [len(frame_names), 3, self.im_height, self.im_width]

            data = []
            for f in frame_names:
                try:
                    img = imread(os.path.join(folder_path, f))
                    img = self.transform(img)
                    data.append(img)
                except:
                    print('SKIPPED A BAD IMAGE FOUND')
            if len(data):
                data = torch.stack(data)
            else:
                data = torch.zeros(video_shape)
            return data

        #===============================================================
        #gallry
        print(self.test_structure['gallery'])
        for vi in self.test_structure['gallery'][0]:
            glr_this_view = []
            for cond in self.test_structure['gallery'][1]:
                for senum in self.test_structure['gallery'][2]:
                    for i, id in enumerate(subjects):
                        glr_this_view.append(read_video(id, cond, senum, vi))
                        # print(vi, i)
            glr_this_view = pad_sequence(glr_this_view, False)
            glr_this_view = glr_this_view[:70] #####################################################
            test_data_glr.append(glr_this_view)
        #===============================================================
        # probe
        print(self.test_structure['probe'])
        for vi in self.test_structure['probe'][0]:
            prb_this_view = []
            for cond in self.test_structure['probe'][1]:
                for senum in self.test_structure['probe'][2]:
                    for i, id in enumerate(subjects):
                        prb_this_view.append(read_video(id, cond, senum, vi))
                        # print(vi, i)
            prb_this_view = pad_sequence(prb_this_view, False)
            prb_this_view = prb_this_view[:70] #####################################################
            test_data_prb.append(prb_this_view)

        return test_data_glr, test_data_prb

    def __len__(self):
        if self.is_train_data:
            return len(self.train_subjects)*110
        else:
            return len(self.test_subjects)*110

