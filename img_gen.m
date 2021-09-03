clear; clc;

file = load('D:\Atik\EEG Data\BCI com III 3A\final.mat');

file = file.final;

dir = 'D:/Atik/EEG Data/BCI com III 3A/img'; fs = 250;

img = {};
for i = 181 : 190
    for j = 1 : size(file{1, 1},1)
        [wt,f] = cwt(file{i, 1}{j, 1},'amor',fs);
        h = figure('Visible', 'off');
        t = 0:numel(file{1, 1}{1, 1})-1;
        hp = pcolor(t,f,abs(wt));
        hp.EdgeColor = 'none';
        set(gca,'xtick',[],'ytick',[],'xticklabel',[],'yticklabel',[]);
        exportgraphics(gca, sprintf('%s/FIG.png', dir));

        img{i,1}{j,1} = imread(sprintf('%s/FIG.png', dir));
        fprintf('Image processed = %d, channel = %d \n', i, j);
    end
end

save(sprintf('%s/img_%d.mat' , dir,i), 'img')
