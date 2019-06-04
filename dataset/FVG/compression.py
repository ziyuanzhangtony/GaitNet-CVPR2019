from zipfile import ZipFile
import os


fvg_structure = {
    'session1': list(range(1,147+1)),
    'session2': list(range(148,226+1)),
    'session3': [1,2,4,7,8,12,13,17,31,40,48,77],
}


in_data_root = '/media/tony/MyBook-MSU-CVLAB/FVG/RAW/'
out_data_root = '/media/tony/Universe/FVG_ZIP/'


for session, sub_ids in fvg_structure.items():
    for sub_id in sub_ids:
        out_file_name = os.path.join(out_data_root, '{:s}_{:03d}.zip'.format(session, sub_id))
        if os.path.exists(out_file_name):
            continue
        with ZipFile(out_file_name, 'w') as zip:
            for vi_idx in range(1,12+1):
                print('{:03d}_{:02d}'.format(sub_id, vi_idx))
                in_folder_name = os.path.join(in_data_root,session,'{:03d}_{:02d}'.format(sub_id, vi_idx))
                frame_names = sorted(os.listdir(in_folder_name))
                frame_names = [f for f in frame_names if f.endswith('.png')]
                # printing the list of all files to be zipped
                # print('Following files will be zipped:')
                # for file_name in frame_names:
                #     print(file_name)

                # writing files to a zipfile

                    # writing each file one by one
                for file in frame_names:
                    zip.write(os.path.join(in_folder_name, file),
                              os.path.join(session,'{:03d}_{:02d}'.format(sub_id, vi_idx), file))