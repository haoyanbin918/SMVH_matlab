function  [XR, YP]=PR_UQ(seti, seedsnew, VSHCs, NewGTX)
% PR  curve
XR=zeros(1,21);% recall
YP=zeros(1,21);%precision
XR(1)=0;
YP(1)=1;
%   load wu_HSVnor;
%   load wu_idxnew;
%   load seedsnew;
%   load NewGTX;
%   load wu_VS_hsv;
%   load wu_stat_end12877;
%   threshold=0;

    gtnum=size(NewGTX{1,seti},1);
    %vistat=wu_stat_end12877(seti,1);
    %viend=wu_stat_end12877(seti,2);
    %TEMP=pdist2(hsv_VS_24dim(vistat:viend,:),hsv_VS_24dim(seedsorig(seti),:));
    
    TEMP=pdist2(VSHCs,  VSHCs(seedsnew(seti),:));
    [SA,IDA]=sort(TEMP);
     SA2=[SA, IDA];
    GotResult=SA2(find(SA2(:,1)<=30),2);
    %GotResult=IDA;
    vinum=size(GotResult,1);
    PR=zeros(vinum,2);
    NewGTXi=NewGTX{1,seti};

    [LA1,IA2]=ismember(GotResult, NewGTXi);
    GR_LA=[GotResult, double(LA1)];
    th=0.05;
    j=1;
for i=1:vinum

    len=length(find(GR_LA(1:i,2)==1)); 
    %grlanum=size(GR_LA,1);
    
    PR(i,1)=len/i;
    PR(i,2)=len/gtnum;
    if PR(i,2)>=th
        j=j+1;
        XR(j)=PR(i,2);
        YP(j)=PR(i,1);

        th=th+0.0499;
    end
   % XR(seti,i)=len/gtnum;
   %YP(seti,i)=len/grlanum;
end
end
        
    
        
        
        
        
    