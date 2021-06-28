clear all;
clc;
close all;

warning('off')

%% load data 
% check motion outliers
thres_mc = 0.15;

load('FD_mats_LR.mat');
FD_mats_LR = FD_mats;
idx1 = find(FD_mats<thres_mc);
load('FD_mats_RL.mat');
FD_mats_RL = FD_mats;
idx2 = find(FD_mats<thres_mc);  

%% Math outliers
load('behav_vec.mat');
behav_vec = behav_vec.MathEI; 

behav_vec = behav_vec(intersect(idx1, idx2));

dropIdx = [];
% using Q value
Q1=prctile(behav_vec,25);
Q3=prctile(behav_vec,75);
IQR = Q3-Q1;
Qmax = Q3 + 1.5*IQR;
Qmin = Q1 - 1.5*IQR;
dropIdx1 = union(find(behav_vec>=Qmax), find(behav_vec<=Qmin));
% using normal distribution
tmp = (behav_vec - mean(behav_vec))./std(behav_vec);
dropIdxMath = union(find(tmp>3), find(tmp<-3)); 
save('dropIdxMath', 'dropIdxMath');

%% Story outliers
load('behav_vec.mat');
behav_vec = behav_vec.StoryEI; 

behav_vec = behav_vec(intersect(idx1, idx2));

dropIdx = [];
% using Q value
Q1=prctile(behav_vec,25);
Q3=prctile(behav_vec,75);
IQR = Q3-Q1;
Qmax = Q3 + 1.5*IQR;
Qmin = Q1 - 1.5*IQR;
dropIdx1 = union(find(behav_vec>=Qmax), find(behav_vec<=Qmin));
% using normal distribution
tmp = (behav_vec - mean(behav_vec))./std(behav_vec);
dropIdxStory = union(find(tmp>3), find(tmp<-3));
save('dropIdxStory', 'dropIdxStory');

