% evaluation main function
% this is for getting the MAP and PR curve
clear

load('traindata_demo\W.mat'); % load learned combination coefficients W and b
load('CC_WEB_VIDEO_data\wu_idxnew.mat'); % load the video's keyframes start and end matrix
load('CC_WEB_VIDEO_data\seedsnew.mat'); % load video seeds
load('CC_WEB_VIDEO_data\NewGTX.mat'); % load groundtruth
load('CC_WEB_VIDEO_data\wu_stat_end12877.mat'); % load the video_start_end index in each set
load('CC_WEB_VIDEO_data\wu_HSVnor.mat'); % load HSV feature matrix n*162
load('CC_WEB_VIDEO_data\wu_LBP_theynor.mat'); % load LBP feature matrix n*256
%========create Hash codes £¨VSHCs£©double =========
wu_HSVLBP_nor = [wu_HSVnor wu_LBP_theynor];
clear wu_HSVnor;
clear wu_LBP_theynor;
VSHCs=CHC(wu_HSVLBP_nor, wu_idxnew, W );
clear wu_HSVLBP_nor;
%================create Hash Codes (BCs)  unit8====================
% Binary Codes
BCs=compactbit(VSHCs>0);
%==================================================

%=======compute MAP===========
MAPALL=zeros(1,24);
TIME=zeros(1,24);
for seti=1:24
     [MAPALL(seti), TIME(seti)] = MAP_CC( seti, seedsnew, BCs, wu_stat_end12877, NewGTX);
end
meanMAP=mean(MAPALL);
meanTIME=mean(TIME);

figure(1)
bar(MAPALL,0.5);
%======end MAP===============

%=======compute PRcurve=========
XR=zeros(24,21);% recall
YP=zeros(24,21);% precision
for seti=1:24
[xr, yp]=PR_CC(seti, seedsnew, VSHCs, wu_stat_end12877, NewGTX);
XR(seti,:)=xr;
YP(seti,:)=yp;
end

xrm=mean(XR);
ypm=mean(YP);
figure(2)
plot(xrm,ypm,'-bo');
%======end PRcurve=============
