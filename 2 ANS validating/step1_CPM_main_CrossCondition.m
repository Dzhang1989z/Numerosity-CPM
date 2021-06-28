clear all;
clc;
close all;

warning('off')

%% load data 
%% load DATA 
load('rest_mats.mat');

load('behav_vec_NC.mat');                       % 161x1
behav_vec = behav_vec_NC;

dropIdx = [];
%% drop outlies data
% using Q value
Q1=prctile(behav_vec,25);
Q3=prctile(behav_vec,75);
IQR = Q3-Q1;
Qmax = Q3 + 1.5*IQR;
Qmin = Q1 - 1.5*IQR;
dropIdx1 = union(find(behav_vec>=Qmax), find(behav_vec<=Qmin));
% using normal distribution
tmp = (behav_vec - mean(behav_vec))./std(behav_vec);
dropIdx2 = union(find(tmp>3), find(tmp<-3));

dropIdx = union(dropIdx1, dropIdx2);

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
    test_mat(isnan(test_mat)) = 0;
    
    pos_mask = AA;
    neg_mask = BB;
   
     %% just mean value and correlation
     tmp = test_mat.*pos_mask;
     behav_pred_pos(leftout) = sum(tmp(:));  
     tmp = test_mat.*neg_mask;
     behav_pred_neg(leftout) = -sum(tmp(:));  
end

%% Compare the predicted performance and ture behavior performance

CCC = [all_behav, behav_pred_pos];

disp('Pearson correlation:')
[R_pos, P_pos] = corr(behav_pred_pos, all_behav)
[R_neg, P_neg] = corr(behav_pred_neg, all_behav)


%% part 1
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
title({'Whole brain network (Positive)', ['Thresh = ' , num2str(thresh),', r = ' num2str(R_pos) ', p = ' num2str(P_pos)]}, 'FontSize', 12);

%% part 1
x = all_behav';
y = behav_pred_neg';

[h, p] = sort(x);
x = x(p);
y = y(p);

figure(2);
plot(x, y, 'bx');
hold on;
[p, s] = polyfit(x, y, 1);
[yfit, dy] = polyconf(p, x, s, 'predopt', 'curve');
plot(x, yfit, 'color', 'r');
plot(x, yfit-dy, 'color', 'r','linestyle',':');
plot(x, yfit+dy, 'color', 'r','linestyle',':');
% xlabel('Observed geometric indtruder accuracy (%)');
% ylabel('Predicted geometric indtruder accuracy (%)');
title({'Whole brain network (Negative)', ['Thresh = ' , num2str(thresh),', r = ' num2str(R_neg) ', p = ' num2str(P_neg)]}, 'FontSize', 12);


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
figure(3);
hist(R_pos_array, 24);
hold on;
plot([R_pos_model R_pos_model], [0 1400], 'r-');
xlabel('R value');
ylabel('Frequency');
title(['Permutation test p = ' num2str(permP)])

% negative part
R_neg_array = [];
for p = 1:10000
    permIdx = randperm(length(all_behav));
    behav_vec_Perm = all_behav(permIdx);
    [R_neg_Perm, drop] = corr(behav_pred_neg, behav_vec_Perm);
    R_neg_array(p) = R_neg_Perm;
end

% math part
R_neg_model = R_neg;
permP = length(find(R_neg_array>R_neg_model))/length(R_neg_array);
figure(4);
hist(R_neg_array, 24);
hold on;
plot([R_neg_model R_neg_model], [0 1400], 'r-');
xlabel('R value');
ylabel('Frequency');
title(['Permutation test p = ' num2str(permP)])



