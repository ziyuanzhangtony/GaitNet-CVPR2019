from zipfile import ZipFile
import os

fvg_structure = {
    'session1': list(range(1,147+1)),
    'session2': list(range(148,226+1)),
    'session3': [1,2,4,7,8,12,13,17,31,40,48,77],
}

in_data_root = '/media/tony/Universe/FVG_ZIP/'
out_data_root = '/media/tony/Universe/FVG_ZIP_TEST/'

def unzip(source_filename, dest_dir):
    with ZipFile(source_filename) as zf:
        zf.extractall(dest_dir)

for session, sub_ids in fvg_structure.items():
    for sub_id in sub_ids:
        print('{:s}_{:03d}.zip'.format(session, sub_id))
        in_file_name = os.path.join(in_data_root, '{:s}_{:03d}.zip'.format(session, sub_id))
        unzip(in_file_name,out_data_root)