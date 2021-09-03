clear; clc;

for i = 10:10:360
    file = load(sprintf(('D:/Atik/EEG Data/BCI com III 3A/img/img_%d.mat'), i));
    file = file.img;
    data(i-9:i, 1) = file(i-9:i, 1);
    clear file;
    fprintf('i = %d \n', i);
end

save(sprintf('%s/data.mat' , dir), 'data', '-v7.3')

data = load('D:\Atik\EEG Data\BCI com III 3A\img\data.mat');
data = data.data;

for i = 1:length(data)
    cell = reshape(data{i,1},[6,10]);
    mat = cell2mat(cell);
    final{i,1} = mat;
    fprintf('i = %d \n', i);
end

for i = 1:length(final)
    resized{i,1} = imresize(final{i, 1},[224 224]);
    fprintf('i = %d \n', i);
end

save(sprintf('%s/resized.mat' , dir), 'resized', '-v7.3')
