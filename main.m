clear; clc;

file = load('D:\Atik\EEG Data\BCI com III 3A\k3b.mat');

a = file.HDR.TRIG;
b = file.s;

sample = {};
for i = 1 : length(a)
    if i == 1
        sample{i,1} = b(1:a(1), :);
    else 
        sample{i,1} = b(a(i -1)+1 : a(i) -1, :);
    end
end

process = {};
for j = 1 : length(sample)
    process{j,1} = sample{j,1}(250*4 + 1 : 250*7, :);
end

all_imf = {};
for i = 1 : length(process)
    imf = {};
    for j = 1 : size(process{1, 1},2)
        imf{j,1} = emd(process{i,1}(:,j));
        fprintf('emd = %d, column = %d \n', i, j)
        all_imf{i,1} = imf;
    end
end

final = {};
for i = 1 : length(all_imf)
    x = {};
    for j = 1 : size(all_imf{1, 1},1)
      x{j,1} = reshape(all_imf{j, 1}{j, 1}(:, 1:5), [], 1);
    end
    final{i,1} = x;
end   
    
% save final.mat final


