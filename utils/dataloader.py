import os
import numpy as np
import torch
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from PIL import Image

def imread(path):
    frame = np.asarray(Image.open(path))
    return frame


class CASIAB(object):

    def __init__(self,
                 is_train_data,
                 data_root,
                 clip_len,
                 im_height,
                 im_width,
                 seed,
                 ):

        np.random.seed(seed)
        self.is_train_data = is_train_data
        self.data_root = data_root
        self.clip_len = clip_len
        self.im_height = im_height
        self.im_width = im_width
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((im_height, im_width)),
            transforms.Pad((2, 4)),
            transforms.Resize((im_height, im_width)),
            transforms.ToTensor()
        ])
        self.video_shape = [self.clip_len, 3, self.im_height, self.im_width]

        subjects = list(range(1, 124 + 1))
        self.train_subjects = subjects[:74]
        self.test_subjects = subjects[74:]
        print('training_subjects', self.train_subjects, len(self.train_subjects))
        print('testing_subjects', self.test_subjects, len(self.test_subjects))

    # def __getitem__(self, index):
    #     shape = [self.clip_len, 3, self.im_height, self.im_width]
    #
    #     def random_chose():
    #         if self.is_train_data:
    #             si_idx = np.random.choice(self.train_subjects)
    #             label = self.train_subjects.index(si_idx)
    #         else:
    #             si_idx = np.random.choice(self.test_subjects)
    #             label = self.test_subjects.index(si_idx)
    #
    #         cond1 = np.random.choice(['nm', 'cl'])
    #         if cond1 == 1:
    #             senum1 = np.random.choice(list(range(1, 2 + 1)))
    #         else:
    #             senum1 = np.random.choice(list(range(1, 2 + 1)))
    #
    #         cond2 = np.random.choice(['nm', 'cl'])
    #         if cond2 == 1:
    #             senum2 = np.random.choice(list(range(1, 2 + 1)))
    #         else:
    #             senum2 = np.random.choice(list(range(1, 2 + 1)))
    #
    #         # view_idx1 = np.random.choice(list(range(0, 180 + 1, 18)))
    #         # view_idx2 = np.random.choice(list(range(0, 180 + 1, 18)))
    #
    #         view_idx1 = np.random.choice([90])
    #         view_idx2 = np.random.choice([90])
    #
    #         return si_idx,(cond1,senum1,view_idx1),(cond2,senum2,view_idx2),label
    #
    #     def random_length(dirt, length):
    #         files = sorted(os.listdir(dirt))
    #         num = len(files)
    #         if num - length < 2:
    #             return 0, 0, []
    #         start = np.random.randint(1, num - length)
    #         end = start + length
    #         return start, end, files
    #
    #     def read_frames(video_in, start, end, files):
    #         ims = np.zeros(shape, np.float32)
    #
    #         for i in range(start, end):
    #             im = imread(os.path.join(video_in, files[i]))
    #             im = self.transform(im)
    #             # im = self.img_random_color(im,ns)
    #             ims[i - start] = im
    #         return ims
    #
    #     si_idx, param1, param2, label = random_chose()
    #     video_in_nm = os.path.join(self.data_root,
    #                                '%03d-%s-%02d-%03d' % (si_idx,
    #                                                        param1[0],
    #                                                        param1[1],
    #                                                        param1[2]))
    #     start_nm, end_nm, files_nm = random_length(video_in_nm, self.sequence_len)
    #
    #     video_in_cl = os.path.join(self.data_root,
    #                                '%03d-%s-%02d-%03d' % (si_idx,
    #                                                        param2[0],
    #                                                        param2[1],
    #                                                        param2[2]))
    #     start_cl, end_cl, files_cl = random_length(video_in_cl, self.sequence_len)
    #
    #     while True:
    #         if start_nm == end_nm == 0 or start_cl == end_cl == 0:
    #             si_idx, param1, param2, label = random_chose()
    #             video_in_nm = os.path.join(self.data_root,
    #                                        '%03d-%s-%02d-%03d' % (si_idx,
    #                                                                param1[0],
    #                                                                param1[1],
    #                                                                param1[2]))
    #             start_nm, end_nm, files_nm = random_length(video_in_nm, self.sequence_len)
    #
    #             video_in_cl = os.path.join(self.data_root,
    #                                        '%03d-%s-%02d-%03d' % (si_idx,
    #                                                                param2[0],
    #                                                                param2[1],
    #                                                                param2[2]))
    #             start_cl, end_cl, files_cl = random_length(video_in_cl, self.sequence_len)
    #         else:
    #             break
    #
    #     TF = np.random.choice([True, False])
    #     if TF:
    #         video_in_mx, start_mx, end_mx, files_mx = \
    #             video_in_nm, start_nm, end_nm, files_nm
    #     else:
    #         video_in_mx, start_mx, end_mx, files_mx = \
    #             video_in_cl, start_cl, end_cl, files_cl
    #
    #     imgs_nm = read_frames(video_in_nm, start_nm, end_nm, files_nm)
    #     imgs_cl = read_frames(video_in_cl, start_cl, end_cl, files_cl)
    #     imgs_mx = read_frames(video_in_mx, start_mx, end_mx, files_mx)
    #     return imgs_nm, imgs_cl,imgs_mx, label

    def __getitem__(self, index):
        """
        chose any two videos for the same subject
        :param index:
        :return:
        """
        data_structure = {
            'nm': list(range(1, 6 + 1)),
            'cl': list(range(1, 2 + 1)),
            'bg': list(range(1, 2 + 1)),
        }

        # 30 x 3 x 32 x 64
        def random_chose():
            if self.is_train_data:
                si_idx = np.random.choice(self.train_subjects)
                label = self.train_subjects.index(si_idx)
            else:

                si_idx = np.random.choice(self.test_subjects)
                label = self.test_subjects.index(si_idx)

            cond1 = np.random.choice(['nm', 'cl'])
            senum1 = np.random.choice(data_structure[cond1])
            cond2 = np.random.choice(['nm', 'cl'])
            senum2 = np.random.choice(data_structure[cond2])

            # view_idx1 = np.random.choice(list(range(0, 180 + 1, 18)))
            # view_idx2 = np.random.choice(list(range(0, 180 + 1, 18)))

            view_idx1 = np.random.choice([90])
            view_idx2 = np.random.choice([90])

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

        return data1, data2, label



    def get_eval_data(self,is_train_data=False):
        if is_train_data is True:
            subjects = self.train_subjects
        else:
            subjects = self.test_subjects

        test_data_glr = []
        test_data_prb = []

        def read_video(si, con, sei, vi):
            folder_path = os.path.join(self.data_root, '%03d-%s-%02d-%03d' % (si, con, sei, vi))
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

        conditions = ['nm','cl']
        ###########################################################
        #gallry
        for vi in [90]:
            glr_this_view = []
            for i, id in enumerate(subjects):
                glr_this_view.append(read_video(id, conditions[0], 1, vi))
                print(vi, i)
            glr_this_view = pad_sequence(glr_this_view, False)
            # glr_this_view = glr_this_view[:100]
            test_data_glr.append(glr_this_view)


        ###########################################################
        # probe
        for vi in [90]:
            prb_this_view = []
            for i, id in enumerate(subjects):
                prb_this_view.append(read_video(id, conditions[1], 1, vi))
                print(vi, i)
            prb_this_view = pad_sequence(prb_this_view, False)
            # prb_this_view = prb_this_view[:100]
            test_data_prb.append(prb_this_view)

        return test_data_glr, test_data_prb

    def __len__(self):
        if self.is_train_data:
            return len(self.train_subjects)*110
        else:
            return len(self.test_subjects)*110


def get_training_batch(data_loader):
    while True:
        for sequences in data_loader:
            batch = [sequence.cuda() for sequence in sequences]
            yield batch