% creat hash codes for key frames
function [HCs] = CHC(X, idX, W) 
% X is keyframes
% idX is the keyframes start and end of each video
% W is transformation matrix which includes W and b
[nhkf, mhkf] = size(X);
nv = size(idX,1);
mw = size(W,2);%
Xone = [X,ones(nhkf,1)];
clear X;
Y = Xone*W;
Z = sigmf(Y,[1 0]);
clear Y;
HCs = zeros(nv,mw);
for i = 1:1:nv
    tempstart = idX(i,1);
    tempend = idX(i,2);
    HCs(i,:) = sum(Z(tempstart:tempend,:),1)/(tempend-tempstart+1);
end
clear Z;
HCs = double(HCs>=0.5);
