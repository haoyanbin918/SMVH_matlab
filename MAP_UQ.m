function [map, time] = MAP_UQ( seti, seedsnew, VSHCs,  NewGTX)
% this function is for evaluate MAP by using Wu'(07) search strategy.
% function main 
% seti is the seti-th class videos
% seed is the seed video's number
% VSHCs is the videos' signature Hash codes
%  load seedsnew;
%  load NewGTX;
%  load wu_VS_hsv;
%  load wu_stat_end12877;

   tic;
    %TEMP=pdist2(VSHCs, VSHCs(seedsnew(seti),:));
    TEMP=hammingDist(VSHCs(seedsnew(seti),:), VSHCs );
    TEMP=double(TEMP');
    %TEMP=pdist2(wu_VS_lbp(vistat:viend,:),wu_VS_lbp(seedsnew(seti),:));
    [SA,IDA]=sort(TEMP);
    toc;
    time=toc;
    
    SA2=[SA, IDA];
    %GotResult=SA2(find(SA2(:,1)<=15),2);
    GotResult=SA2(:,2);
    %NDVS{1,seti}=GotResult;
    NewGTXi=NewGTX{1,seti};
    
   % NewGTXi=newGroundTruth{1,seti}-vistat+1;
    [LA1,IA2]=ismember(GotResult, NewGTXi);
    GR_LA=[GotResult, double(LA1)];
    %NDVSh{1,seti}=GR_LA;
    SNUM=size(GR_LA,1);
    tempnum=0;
    maptemp=0;
    %seedsnew(seti)
    %AA=sort(GR_LA(:,2),'descend');
    %len=length(find(GR_LA(:,2)==1));
    %NDVnumberh(1,seti)=len;
    for im=1:1:SNUM
        
        if GR_LA(im,2)==1
        tempnum=tempnum+1;
        maptemp=tempnum/im+maptemp;
        
        end
    end
    allNDVs=size(NewGTX{1,seti},1);  
    map=maptemp/allNDVs;
end


