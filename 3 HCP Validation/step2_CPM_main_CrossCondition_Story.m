clear all;
clc;
close all;

warning('off')

%% load data 
thres_mc = 0.15;

load('FD_mats_LR.mat');
FD_mats_LR = FD_mats;
idx1 = find(FD_mats<thres_mc);
load('FD_mats_RL.mat');
FD_mats_RL = FD_mats;
idx2 = find(FD_mats<thres_mc);  

load('rest_mats_LR_preproc.mat');
rest_mats_LR = rest_mats;

load('rest_mats_RL_preproc.mat');
rest_mats_RL = rest_mats;
rest_mats = (rest_mats_LR + rest_mats_RL)/2;
% rest_mats = (rest_mats_RL + rest_mats_RL)/2;

load('behav_vec.mat');
behav_vec = behav_vec.StoryEI; 

behav_vec = behav_vec(intersect(idx1, idx2));
rest_mats = rest_mats(:,:,intersect(idx1, idx2));
AA = intersect(idx1, idx2);

dropIdx = [];
%% drop outlies data
load('dropIdxStory.mat');
load('dropIdxMath.mat');
dropIdx = union(dropIdxMath, dropIdxStory);
behav_vec(dropIdx) = [];

%%
all_mats = rest_mats;   % FC matrix, d
all_mats(:,:,dropIdx) = [];
for p = 1:size(all_mats,3)
    for q = 1:size(all_mats,1)
        all_mats(q,q, p) = 0;
    end   
end
all_mats = 0.5*log((1+all_mats)./(1-all_mats));
all_behav = behav_vec;  % behavior vector, 263x1

%% 
% threshold for FC selection
thresh = 0.01;

%% Define parameters
no_sub = size(all_mats,3);
no_node = size(all_mats,1);

behav_pred_pos = zeros(no_sub,1);
behav_pred_neg = zeros(no_sub,1);

whether_standard = 0;

load('pos_mat_NC001.mat');  % 159x159x141
load('neg_mat_NC001.mat');  % 159x159x141

dropIdx = [100:133, 236:268];
IdxROIs268Left = setdiff(1:268,dropIdx);

OverAllThres = 1;
A = mean(pos_mat_save, 3);
AA = zeros(268, 268);
AA(IdxROIs268Left, IdxROIs268Left) = A;
AA(find(AA<OverAllThres)) = 0;
AA(find(AA>=OverAllThres)) = 1;
B = mean(neg_mat_save, 3);
BB = zeros(268, 268);
BB(IdxROIs268Left, IdxROIs268Left) = B;
BB(find(BB<OverAllThres)) = 0;
BB(find(BB>=OverAllThres)) = 1;

%% LOOCV
for leftout = 1:no_sub 
    disp(['Leaving out subject #' num2str(leftout) '.']);
    % leave out one subject
    test_mat = all_mats(:,:,leftout);
    pos_mask = AA;
    neg_mask = BB;
    
    %% calculate the FC sum of posives edges and negative edges
     %% just mean value and correlation
        tmp = test_mat.*pos_mask;
        behav_pred_pos(leftout) = sum(tmp(:));  
        tmp = test_mat.*neg_mask;
        behav_pred_neg(leftout) = -sum(tmp(:));  
end

%% Compare the predicted performance and ture behavior performance

disp('Pearson correlation:')
[R_pos, P_pos] = corr(behav_pred_pos, all_behav)

%% plot part
x = all_behav';
y = behav_pred_pos';

[h, p] = sort(x);
x = x(p);
y = y(p);

figure(1);
plot(x, y, 'bx');
hold on;
[p, s] = polyfit(x, y, 1);
[yfit, dy] = polyconf(p, x, s, 'predopt', 'curve');
plot(x, yfit, 'color', 'r');
plot(x, yfit-dy, 'color', 'r','linestyle',':');
plot(x, yfit+dy, 'color', 'r','linestyle',':');

ylabel('Numerosity network strength');
title({'HCP math task', ['r = ' num2str(R_pos) ', p = ' num2str(P_pos)]}, 'FontSize', 12);

%% ÖÃ»»¼ìÑé
R_pos_array = [];
for p = 1:10000
    permIdx = randperm(length(all_behav));
    behav_vec_Perm = all_behav(permIdx);
    [R_pos_Perm, drop] = corr(behav_pred_pos, behav_vec_Perm);
    R_pos_array(p) = R_pos_Perm;
end

% math part
R_pos_model = R_pos;
permP = length(find(R_pos_array>R_pos_model))/length(R_pos_array);
figure(2);
hist(R_pos_array, 24);
hold on;
plot([R_pos_model R_pos_model], [0 1400], 'r-');
xlabel('R value');
ylabel('Frequency');
title(['Permutation test p = ' num2str(permP)]);