import os
input = '/home/tony/Documents/CASIA-B-/SEG'
output = '/home/tony/Documents/CASIA-B-/choose'
from shutil import copyfile

for si in range(1,125):
    for cond in ['nm','cl','bg']:
        sei = 1
        for vi in [90]:
            video_name = input+"/{:03d}-{:s}-{:02d}-{:03d}".format(si,cond,sei,vi)
            files = os.listdir(video_name)
            thefile = os.path.join(video_name,files[0])
            target_name = output + "/{:03d}-{:s}-{:02d}-{:03d}-{:s}.png".format(si,cond,sei,vi,files[0])
            copyfile(thefile, target_name)
            print(video_name)