import os
import random

import numpy as np
import torch
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from PIL import Image

from utils.compute import *

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
            transforms.ToTensor(),
        ])

        subjects = list(range(1, 124 + 1)) # 124 subjects
        # random.Random(1).shuffle(subjects)
        self.train_subjects = subjects[:opt.num_train]
        self.test_subjects = subjects[opt.num_train:]
        self.step = 0

        print('Number of Training_Subjects', self.train_subjects, len(self.train_subjects))
        print('testing_subjects', self.test_subjects, len(self.test_subjects))

    # def __getitem__(self, index): #new
    #
    #     # video_shape = [self.clip_len, 3, self.im_height, self.im_width]
    #     # conditions = self.train_structure['clip1'][1].union(self.train_structure['clip2'][1])
    #
    #     # 30 x 3 x 32 x 64
    #     def random_chose():
    #         if self.is_train_data:
    #             si_idx = np.random.choice(self.train_subjects)
    #             label = self.train_subjects.index(si_idx)
    #         else:
    #             si_idx = np.random.choice(self.test_subjects)
    #             label = self.test_subjects.index(si_idx)
    #
    #         view_idx1 = np.random.choice(self.train_structure['clip1'][0])
    #         cond1 = np.random.choice(list(self.train_structure['clip1'][1].keys()))
    #         senum1 = np.random.choice(self.train_structure['clip1'][1][cond1])
    #
    #         view_idx2 = np.random.choice(self.train_structure['clip2'][0])
    #         cond2 = np.random.choice(list(self.train_structure['clip2'][1].keys()))
    #         senum2 = np.random.choice(self.train_structure['clip2'][1][cond2])
    #
    #         return si_idx, (cond1, senum1, view_idx1), (cond2, senum2, view_idx2), label
    #
    #     def random_length(dirt, length):
    #         files = sorted(os.listdir(dirt))
    #         num = len(files)
    #         if num - length < 2:
    #             return None
    #         start = np.random.randint(1, num - length)
    #         end = start + length
    #         return files[start:end]
    #
    #     def read_video(frames_pth, file_names):
    #         # T = transforms.Compose([
    #         #     transforms.ToPILImage(),
    #         #     transforms.ColorJitter(brightness=(0,1), contrast=(0,1), saturation=(0,1), hue=(-0.5,0.5)),
    #         #     transforms.ToTensor(),
    #         # ])
    #         # frames = np.zeros(self.im_shape, np.float32)
    #         frames = []
    #         for f in file_names:
    #             frame = np.asarray(Image.open(os.path.join(frames_pth, f)))
    #             frame = self.transform(frame)
    #             # frame = T(frame)
    #             frames.append(frame)
    #         frames = torch.stack(frames)
    #         return frames
    #
    #     si, param1, param2, label = random_chose()
    #     frames_pth1 = os.path.join(self.data_root,
    #                                '%03d-%s-%02d-%03d' % (si,
    #                                                       param1[0],
    #                                                       param1[1],
    #                                                       param1[2]))
    #     frames_pth2 = os.path.join(self.data_root,
    #                                '%03d-%s-%02d-%03d' % (si,
    #                                                       param2[0],
    #                                                       param2[1],
    #                                                       param2[2]))
    #     file_names1 = random_length(frames_pth1, self.clip_len)
    #     file_names2 = random_length(frames_pth2, self.clip_len)
    #
    #     while True:
    #         if file_names1 == None or file_names2 == None:
    #             si, param1, param2, label = random_chose()
    #             frames_pth1 = os.path.join(self.data_root,
    #                                        '%03d-%s-%02d-%03d' % (si,
    #                                                               param1[0],
    #                                                               param1[1],
    #                                                               param1[2]))
    #             frames_pth2 = os.path.join(self.data_root,
    #                                        '%03d-%s-%02d-%03d' % (si,
    #                                                               param2[0],
    #                                                               param2[1],
    #                                                               param2[2]))
    #             file_names1 = random_length(frames_pth1, self.clip_len)
    #             file_names2 = random_length(frames_pth2, self.clip_len)
    #         else:
    #             break
    #
    #     data1 = read_video(frames_pth1, file_names1)
    #     data2 = read_video(frames_pth2, file_names2)
    #
    #     if np.random.choice([True, False]):
    #         data3 = data1.clone()
    #     else:
    #         data3 = data2.clone()
    #
    #     return data1, data2, data3, label




    def __getitem__(self, index): #new

        # video_shape = [self.clip_len, 3, self.im_height, self.im_width]
        # conditions = self.train_structure['clip1'][1].union(self.train_structure['clip2'][1])

        # 30 x 3 x 32 x 64
        def random_chose(existing_si=False):
            if existing_si is not False:
                if self.is_train_data:
                    train_subjects_copy = self.train_subjects[:]
                    train_subjects_copy.remove(existing_si)
                    si_idx = np.random.choice(train_subjects_copy)
                    label = self.train_subjects.index(si_idx) # find label from original
                else:
                    testing_subjects_copy = self.test_subjects[:]
                    testing_subjects_copy.remove(existing_si)
                    si_idx = np.random.choice(testing_subjects_copy)
                    label = self.test_subjects.index(si_idx) # find label from original
            else:
                if self.is_train_data:
                    si_idx = np.random.choice(self.train_subjects)
                    label = self.train_subjects.index(si_idx)
                else:
                    si_idx = np.random.choice(self.test_subjects)
                    label = self.test_subjects.index(si_idx)

            view_idx1 = np.random.choice(self.train_structure['clip1'][0])
            cond1 = np.random.choice(list(self.train_structure['clip1'][1].keys()))
            senum1 = np.random.choice(self.train_structure['clip1'][1][cond1])

            view_idx2 = np.random.choice(self.train_structure['clip2'][0])
            cond2 = np.random.choice(list(self.train_structure['clip2'][1].keys()))
            senum2 = np.random.choice(self.train_structure['clip2'][1][cond2])

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
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.im_height, self.im_width)),

                # transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                # transforms.Pad((2, 4)),
                # transforms.Resize((self.im_height, self.im_width)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

            frames = []
            for f in file_names:
                frame = np.asarray(Image.open(os.path.join(frames_pth, f)))
                frame = transform(frame)
                # frame = T(frame)
                frames.append(frame)
            frames = torch.stack(frames)
            return frames

        si_a, param1_a, param2_a, label_a = random_chose()
        si_b, param1_b, _, label_b = random_chose(si_a)

        frames_pth1 = os.path.join(self.data_root,
                                   '%03d-%s-%02d-%03d' % (si_a,
                                                          param1_a[0],
                                                          param1_a[1],
                                                          param1_a[2]))
        frames_pth2 = os.path.join(self.data_root,
                                   '%03d-%s-%02d-%03d' % (si_a,
                                                          param2_a[0],
                                                          param2_a[1],
                                                          param2_a[2]))
        frames_pth3 = os.path.join(self.data_root,
                                   '%03d-%s-%02d-%03d' % (si_b,
                                                          param1_b[0],
                                                          param1_b[1],
                                                          param1_b[2]))

        file_names1 = random_length(frames_pth1, self.clip_len)
        file_names2 = random_length(frames_pth2, self.clip_len)
        file_names3 = random_length(frames_pth3, self.clip_len)


        while True:
            if file_names1 is None or file_names2 is None or file_names3 is None:
                si_a, param1_a, param2_a, label_a = random_chose()
                si_b, param1_b, _, label_b = random_chose(si_a)

                frames_pth1 = os.path.join(self.data_root,
                                           '%03d-%s-%02d-%03d' % (si_a,
                                                                  param1_a[0],
                                                                  param1_a[1],
                                                                  param1_a[2]))
                frames_pth2 = os.path.join(self.data_root,
                                           '%03d-%s-%02d-%03d' % (si_a,
                                                                  param2_a[0],
                                                                  param2_a[1],
                                                                  param2_a[2]))
                frames_pth3 = os.path.join(self.data_root,
                                           '%03d-%s-%02d-%03d' % (si_b,
                                                                  param1_b[0],
                                                                  param1_b[1],
                                                                  param1_b[2]))

                file_names1 = random_length(frames_pth1, self.clip_len)
                file_names2 = random_length(frames_pth2, self.clip_len)
                file_names3 = random_length(frames_pth3, self.clip_len)
            else:
                break

        data1 = read_video(frames_pth1, file_names1)
        data2 = read_video(frames_pth2, file_names2)
        data3 = read_video(frames_pth3, file_names3)

        return data1, data2, label_a, data3, label_b

    def get_analogy_format(self, row=None, col=None):
        if row == None and col ==None:
            # subject:[frame,view-angle,condition]
            # row = [
            #     [21, 93, 90, 'nm'],
            #     [21, 47, 90, 'cl'],
            #
            #     [1, 60, 90, 'nm'],
            #     [1, 60, 90, 'cl'],
            #
            #     [3, 55, 90, 'nm'],
            #     [3, 55, 90, 'cl'],
            #
            #     [4, 80, 90, 'nm'],
            #     [4, 80, 90, 'cl'],
            #
            #     [12, 93, 90, 'nm'],
            #     [12, 93, 90, 'cl'],
            # ]
            # col = [
            #     [1, 60, 90, 'nm'],
            #     [2, 55, 90, 'nm'],
            #     [3, 55, 90, 'nm'],
            #     [4, 80, 90, 'nm'],
            #     [12, 93, 90, 'nm'],
            # ]

            # row = [
            #     [75, 59, 0, 'nm'],
            #     [76, 64, 0, 'nm'],
            #     # [3, 55, 90, 'nm'],
            #     # [4, 80, 90, 'nm'],
            #     [77, 84, 0, 'nm'],
            #        ]
            # col = [
            #     # [1, 60, 90, 'nm'],
            #     # [2, 55, 90, 'nm'],
            #     # [3, 55, 90, 'nm'],
            #     # [4, 80, 90, 'nm'],
            #     # [6, 55, 90, 'nm'],
            #     [79, 33, 0, 'nm'],
            #     [79, 33, 0, 'cl'],
            #     [83, 89, 0, 'nm'],
            #        ]


            row = [
                # [75, 93, 90, 'nm'],
                # [75, 47, 90, 'cl'],
                # [116, 93, 90, 'nm'],
                # [116, 93, 90, 'cl'],
                # [109, 93, 90, 'nm'],
                # [109, 93, 90, 'cl'],
                # [103, 65, 90, 'nm'],
                # [103, 65, 90, 'cl'],
                # [100, 47, 90, 'nm'],
                # [100, 47, 90, 'cl'],
                [117, 93, 90, 'nm'],
                [117, 93, 90, 'cl'],
                [118, 93, 90, 'nm'],
                [118, 93, 90, 'cl'],
                [97, 93, 90, 'nm'],
                [97, 93, 90, 'cl'],
                [93, 93, 90, 'nm'],
                [93, 93, 90, 'cl'],
                [91, 47, 90, 'nm'],
                [91, 47, 90, 'cl'],
            ]
            col = [

                [119, 93, 90, 'nm'],
                [118, 93, 90, 'cl'],
                [97, 93, 90, 'nm'],
                [93, 93, 90, 'cl'],
            ]


        print('Loading analogy format')


        data_row = []
        data_col = []

        for para in row:
            im_path = os.path.join(self.data_root, '%03d-%s-%02d-%03d' % (para[0], para[3], 1, para[2]),'%04d.png'%para[1])
            im = imread(im_path)
            im = self.transform(im)
            data_row.append(im)

        for para in col:
            im_path = os.path.join(self.data_root, '%03d-%s-%02d-%03d' % (para[0], para[3], 1, para[2]),'%04d.png'%para[1])
            im = imread(im_path)
            im = self.transform(im)
            data_col.append(im)

        return data_row, data_col


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
            # frame_names = os.listdir(folder_path)
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

        gt = []
        #===============================================================
        #gallry
        print(self.test_structure['gallery'])
        for vi in self.test_structure['gallery'][0]:
            glr_this_view = []
            for i, id in enumerate(subjects):
                for cond in self.test_structure['gallery'][1]:
                    for senum in self.test_structure['gallery'][2]:
                        glr_this_view.append(read_video(id, cond, senum, vi))
                    # print(vi, i)
            glr_this_view = pad_sequence(glr_this_view, False)
            glr_this_view = glr_this_view[:50] #####################################################
            test_data_glr.append(glr_this_view)
        #===============================================================
        # probe
        print(self.test_structure['probe'])
        for vi in self.test_structure['probe'][0]:
            prb_this_view = []
            for i, id in enumerate(subjects):
                for cond in self.test_structure['probe'][1]:
                    for senum in self.test_structure['probe'][2]:
                        prb_this_view.append(read_video(id, cond, senum, vi))
                        gt.append(i)
                        # print(vi, i)
            prb_this_view = pad_sequence(prb_this_view, False)
            prb_this_view = prb_this_view[:50] #####################################################
            test_data_prb.append(prb_this_view)

        return test_data_glr, test_data_prb

    def get_tsne_format(self):
        print('Loading tsne format')

        print(self.is_train_data)
        if self.is_train_data:
            subjects = self.train_subjects
        else:
            subjects = self.test_subjects

        video = []
        label = []

        def read_video(si, con, sei, vi):
            folder_path = os.path.join(self.data_root, '%03d-%s-%02d-%03d' % (si, con, sei, vi))
            frame_names = sorted(os.listdir(folder_path))
            # frame_names = os.listdir(folder_path)
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
        for si in range(105, 105+10):
            for cond in ['nm', 'cl']:
                sei = 1
                for vi in [90]:
                    video.append(read_video(si, cond, sei, vi))
                    label.extend([si-105])
        label = label*30
        video = pad_sequence(video,False)
        video = video[:30]  #####################################################
        label = torch.tensor(label)

        return video, label


    def __len__(self):
        if self.is_train_data:
            return len(self.train_subjects)*110
        else:
            return len(self.test_subjects)*110




class FVG(object):

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
            # transforms.Pad((2, 4)),
            # transforms.Resize((self.im_height, self.im_width)),
            transforms.ToTensor(),
        ])

        subjects = list(range(1, 226 + 1))
        random.Random(1).shuffle(subjects)
        self.train_subjects = subjects[:opt.num_train]
        self.test_subjects = subjects[opt.num_train:]

        print('Number of Training_Subjects', self.train_subjects, len(self.train_subjects))
        print('testing_subjects', self.test_subjects, len(self.test_subjects))

    def __getitem__(self, index):
        """
        chose any two videos for the same subject
        :param index:
        :return:
        """

        # 30 x 3 x 32 x 64
        def random_chose():
            if self.is_train_data:
                si_idx = np.random.choice(self.train_subjects)
                label = self.train_subjects.index(si_idx)
            else:
                si_idx = np.random.choice(self.test_subjects)
                label = self.test_subjects.index(si_idx)

            if si_idx in list(range(1, 147 + 1)):
                if si_idx in [1, 2, 4, 7, 8, 12, 13, 17, 31, 40, 48, 77]:
                    reading_dir1 = random.choice(['session1', 'session3'])
                    reading_dir2 = random.choice(['session1', 'session3'])
                else:
                    reading_dir1 = 'session1'
                    reading_dir2 = 'session1'
            else:
                reading_dir1 = 'session2'
                reading_dir2 = 'session2'

            vi_idx1 = np.random.choice(self.train_structure[reading_dir1])
            vi_idx2 = np.random.choice(self.train_structure[reading_dir2])

            return si_idx, (reading_dir1, vi_idx1), (reading_dir2, vi_idx2), label

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
        frames_pth1 = os.path.join(self.data_root, param1[0], '%03d_%02d' % (si, param1[1]))
        frames_pth2 = os.path.join(self.data_root, param2[0], '%03d_%02d' % (si, param2[1]))
        file_names1 = random_length(frames_pth1, self.clip_len)
        file_names2 = random_length(frames_pth2, self.clip_len)

        while True:
            if file_names1 == None or file_names2 == None:
                si, param1, param2, label = random_chose()
                frames_pth1 = os.path.join(self.data_root, param1[0], '%03d_%02d' % (si, param1[1]))
                frames_pth2 = os.path.join(self.data_root, param2[0], '%03d_%02d' % (si, param2[1]))
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


    def get_eval_format(self, gallery, probe_s1, probe_s2):
        if self.is_train_data is True:
            subjects = self.train_subjects
        else:
            subjects = self.test_subjects

        gt = []

        def read_frames(si_idx, vi_idx):
            if si_idx in list(range(1, 147 + 1)):
                reading_dir = 'session1'
            else:
                reading_dir = 'session2'

            folder_path = os.path.join(self.data_root, reading_dir, '%03d_%02d' % (si_idx, vi_idx))

            frame_names = sorted(os.listdir(folder_path))
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

        test_data_glr = []
        test_data_prb = []

        for i, id in enumerate(subjects):
            test_data_glr.append(read_frames(id, gallery))
            if id in list(range(1, 147 + 1)):
                probes = probe_s1
            else:
                probes = probe_s2
            gt.append(len(probes))
            for j in probes:
                test_data_prb.append(read_frames(id, j))
            print('Training data?:', self.is_train_data, '. Reading', i, 'th subject:', id)

        test_data_glr = pad_sequence(test_data_glr)
        test_data_glr = test_data_glr[:70]
        test_data_prb = pad_sequence(test_data_prb)
        test_data_prb = test_data_prb[:70]
        return test_data_glr, \
               test_data_prb, \
               gt


    def get_eval_format_all(self):
        if self.is_train_data is True:
            subjects = self.train_subjects
        else:
            subjects = self.test_subjects

        gt = []

        def read_frames(si_idx, vi_idx):
            if si_idx in list(range(1, 147 + 1)):
                reading_dir = 'session1'
            else:
                reading_dir = 'session2'

            folder_path = os.path.join(self.data_root, reading_dir, '%03d_%02d' % (si_idx, vi_idx))

            frame_names = sorted(os.listdir(folder_path))
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

        test_data_glr = []
        test_data_prb = []

        for i, id in enumerate(subjects):
            test_data_glr.append(read_frames(id, 2))
            if id in list(range(1, 147 + 1)):
                session = ['session1']
                if id in [1, 2, 4, 7, 8, 12, 13, 17, 31, 40, 48, 77]:
                    session = ['session1', 'session3']
            else:
                session = ['session2']
            probe_cnt = 0
            for se in session:
                if se == 'session1' or se == 'session2':
                    probes = [1]+list(range(3,12+1))
                else:
                    probes = list(range(1, 12 + 1))
                for j in probes:
                    test_data_prb.append(read_frames(id, j))
                    probe_cnt+=1
            gt.append(probe_cnt)

            print('Training data?:', self.is_train_data, '. Reading', i, 'th subject:', id)

        test_data_glr = pad_sequence(test_data_glr)
        test_data_glr = test_data_glr[:70]
        test_data_prb = pad_sequence(test_data_prb)
        test_data_prb = test_data_prb[:70]
        return test_data_glr, \
               test_data_prb, \
               gt

    def get_eval_format_time(self):

        subjects_s3 = [1, 2, 4, 7, 8, 12, 13, 17, 31, 40, 48, 77]
        subjects = set(self.test_subjects) & set(subjects_s3)

        gt = []

        def read_frames(si_idx, vi_idx, session):
            reading_dir = session
            folder_path = os.path.join(self.data_root, reading_dir, '%03d_%02d' % (si_idx, vi_idx))

            frame_names = sorted(os.listdir(folder_path))
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

        test_data_glr = []
        test_data_prb = []




        for i, id in enumerate(subjects):
            test_data_glr.append(read_frames(id, 2, 'session1'))
            probes = [2]
            gt.append(len(probes))
            for j in probes:
                test_data_prb.append(read_frames(id, j, 'session3'))
            print('Training data?:', self.is_train_data, '. Reading', i, 'th subject:', id)

        test_data_glr = pad_sequence(test_data_glr)
        test_data_glr = test_data_glr[:70]
        test_data_prb = pad_sequence(test_data_prb)
        test_data_prb = test_data_prb[:70]
        return test_data_glr, \
               test_data_prb, \
               gt


    def __len__(self):
        if self.is_train_data:
            return len(self.train_subjects) * 12
        else:
            return len(self.test_subjects) * 12
