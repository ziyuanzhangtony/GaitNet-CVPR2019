listing = {};

folder_listing = dir('FORD_MSU_png');
folder_listing(1:2)=[];

for(i = 1:length(folder_listing))
    
    folder_dir = [folder_listing(i).folder '/' folder_listing(i).name];
    files_listing = dir(folder_dir);
    files_listing(1:2)=[];
    
    for(j = 1:length(files_listing))
        file_dir = [files_listing(j).folder '/' files_listing(j).name];
        listing = [listing;file_dir];
    end
end

%%

fileID = fopen('img-list.txt','w');
for(i = 1:length(listing))
    fprintf(fileID,[listing{i} '\n']);
end
fclose(fileID);