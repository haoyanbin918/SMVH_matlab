% This is the main function of SMVH
% This function is part of SMVH Implementation for paper:"Stochastic Multi-view Hashing for
% Large-scale Near-duplicate Video Retrieval"
% You are free to use, change, or redistribute this code in any way you
% want for non-commercial purposes. However, it is appreciated if you 
% maintain the name of the original author.
%
% (C) Yanbin Hao, Hefei University of Technology

clear
%% train 
% suppose n training keyframes
load('traindata_demo\tradata_hsv.mat'); % tradata_hsv is the HSV feature matrix, with the size of n*d_hsv, while d_hsv is the dimensionality of HSV.
load('traindata_demo\tradata_lbp.mat'); % tradata_lbp is the LBP feature matrix, with the size of n*d_lbp, while d_lbp is the dimensionality of LBP.

HSVLBP=[tradata_hsv, tradata_lbp];

clear tradata_lbp;
clear tradata_hsv;

%====== compute conditonal probability matrix P ===============
P_hsv = X2Psig(tradata_hsv,15);
P_lbp = X2Psig(tradata_lbp,15);
%load('traindata_demo\P_hsv.mat'); 
%load('traindata_demo\P_lbp.mat'); 
load('traindata_demo\Lnor.mat'); % Lnor is the normalized group matrix, with the size of n*n and each element of the main diagonal is zero.
load('traindata_demo\SPVnor.mat'); % SPVnor is the normalized supervised proximity matrix, with the size of n*n and each element of the main diagonal is zero.

PLS=0.4*P_hsv +0.3*P_lbp+ 0.01*Lnor+0.29*SPVnor;

clear P_hsv;
clear P_lbp;
clear Lnor;
clear SPVnor;

% W: [W; b] combination coefficients W and bias term b 
% Z relaxed hash codes
[W, Z] = SMVH_graddesc_stan(HSVLBP,PLS,320);

clear PLS;
clear HSVLBP;
save('traindata_demo\W.mat', 'W');
clear W;
clear Z;

%% evaluation
% MAP and PR curve
evaluation_CC_main;








