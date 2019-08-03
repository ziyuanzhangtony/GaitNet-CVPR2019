import os
from imageio import imread

seg_out_data_root = '/home/tony/Documents/CASIA-B-/SEG/'
sil_out_data_root = '/home/tony/Documents/CASIA-B-/SIL/'
crop_out_data_root = '/home/tony/Documents/CASIA-B-/CROP/'
# out_data_root = '/media/tony/MyBook-MSU-CVLAB/FVG-NEW/SEG/'

cb_structure = {
    'nm': list(range(1,6+1)),
    'cl': list(range(1,2+1)),
    'bg': list(range(1,2+1)),
}

def delete_files(path):
    files = sorted(os.listdir(path))
    for f in files:
        file_name = os.path.join(path, f)

        img = imread(file_name)
        if int(img.shape[0]/2) != img.shape[1]:
            # print(file_name)
            os.remove(file_name)

for sub_id in range(1,124+1):
    for cond,vi_idxs in cb_structure.items():
        for vi_idx in vi_idxs:
            for view_angle in range(0,180+1,18):
                video_name = '{:03d}-{:s}-{:02d}-{:03d}'.format(sub_id, cond, vi_idx, view_angle)
                # print(video_name)
                video_path = os.path.join(seg_out_data_root,video_name)
                # print(video_path)
                seg_out_folder_name = os.path.join(seg_out_data_root,video_name)
                sil_out_folder_name = os.path.join(sil_out_data_root, video_name)
                crop_out_folder_name = os.path.join(crop_out_data_root, video_name)
                print(seg_out_folder_name)
                delete_files(seg_out_folder_name)
                delete_files(sil_out_folder_name)
                delete_files(crop_out_folder_name)
