%%
IPUT = 'FVG-SLH-S3';
OPUT = 'FVG-GEI-S3';
folderdirs = dir(IPUT);folderdirs(1:2) = [];
%% segmentation

for i =1:length(folderdirs)
    sildr = [IPUT '/' folderdirs(i).name];
    write_to = [OPUT '/' folderdirs(i).name '.png'];
%     mkdir(write_to)
    
    files = dir([sildr '/*.png']);
    GEI=zeros(256,128);
    for j = 1:length(files)
        sil = imread([sildr '/' files(j).name]);
        sil = im2double(sil);
        GEI=GEI+sil;
    end
    GEI = GEI/length(files);
    imwrite(GEI,write_to)
    write_to
%     break;
end




 
