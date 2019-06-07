ipath = 'CB-RGB-BS/001-2-02-090';
opath = 'imgs';

files = dir([ipath '/*.png']);

for i = 1:length(files)
    img = imread([ipath '/' files(i).name]);
    new_img = white_balance(img);
    imwrite(new_img,[opath '/' files(i).name]);
end

