clc
clear all
close all
%M=csvread("WS_10m.csv",1,3);
%M=csvread("WS_hr.csv",2,3,[2 3 3000 3]);
M=csvread("WS_KFUPM_10m_2015.csv",1,2);
% [10 20 30 40 60 80 100 120 140 160 180]
% [10 20 30 40 x50 60 x70 80 x90 100 x110 120 x130 140 x150 160 x170 180]
%M=M(:,[2 4 5 6 7 8 10 

days=360;
%numdat=6*24*days;
inputsize=4;
%M=M(1:(numdat),:);
N=length(M);

for k=1:11
for i=1:N  
    if M(i,k)> 20 %CLEAN THE DATA if 9999, then replace with previous value
        M(i,k)=M(i-1,k);
    elseif M(i,k)<= 0
    M(i,k)=M(i-1,k);
    end
end
end

M=fliplr(M);

ii=1;
for i=1:N
    diff0=M(i,2:11)-M(i,1:10);
    lt0=sum(find(diff0<0.1));
    if lt0==0 
        MN(ii,:)=M(i,:);
        ii=ii+1;
    end
end

mt15=find(MN(:,6)<=15);   
M=MN(mt15,:);
mt10=find(M(:,3)<=10);   
M=M(mt10,:);

M=[M(:,[1 2 3 4]) (M(:,4)+M(:,5))/2 M(:,5) (M(:,5)+M(:,6))/2 M(:,6) (M(:,6)+M(:,7))/2 M(:,7) (M(:,7)+M(:,8))/2 M(:,8) (M(:,8)+M(:,9))/2 M(:,9) (M(:,9)+M(:,10))/2 M(:,10) (M(:,10)+M(:,11))/2 M(:,11)];
% [10 20 30 40 60 80 100 120 140 160 180]
% [10 20 30 40 x50 60 x70 80 x90 100 x110 120 x130 140 x150 160 x170 180]
%

perc=100;
numdat=length(M);

%R=6; % Every 6 makes an hour
%mm=floor(N/R);
%for i=1:mm
%    j=(i-1)*R+1;
%    MD(i,1)=mean(M(j:j+R-1));
%end

trainingnum=floor(0.8*numdat); % Num of training samples
maxx=max(max(M(1:trainingnum,1:inputsize)));
training=M(1:trainingnum,:);

series=training/maxx;
datasize=size(series);
nex=1;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing
Nhid=5;
Rrr=0.0000001;
testing=M((trainingnum+1):end,:);

seriesT=testing/maxx;
%numdata=max(datasize)-(inputsize+ahead-1);
testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

P50 = traininginput';
Y50 = trainingtarget';
Ptest50 = testinginput';
Ytest50 = testingtarget';
testingtarget50=Ytest50'*maxx;
%
%Create NN

%outval = netGAM(P);

trainingtargetmax=trainingtarget*maxx;

height50=[10 20 30 40 50];
rang50=[0 13];
rl50=[1:13];
% GAM WSE

GAMP50 = traininginput';
GAMY50 = trainingtarget';
GAMPtest50 = testinginput';
GAMYtest50 = testingtarget';
GAMtestingtarget50=GAMYtest50'*maxx;

%netGAM = fitrGAM(GAMP50',GAMY50,'OptimizeHyperparameters','epsilon');
netGAM = fitrgam(GAMP50',GAMY50);
outval = (predict(netGAM,GAMP50'))';

outvalmax=outval*maxx;
GAMOutf50train=outvalmax';
%mse(GAMOutf50train,GAMY50*maxx)
%outvaltest=(sigmoid(Ww*GAMPtest50)'*Beta)';
outvaltest=(predict(netGAM,GAMPtest50'))';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget50;
GAMOutf50=outvaltestmax;
GAMmsetest50=mse(GAMOutf50,testingtarget50);
GAMOut=GAMOutf50;
    GAMmsetest=mse(GAMOut,testingtargetmax);
    GAMrmsetest=rmse(GAMOut,testingtargetmax);
    GAMmaetest=mae(GAMOut,testingtargetmax);
    GAMmbetest=mbe(GAMOut,testingtargetmax);
    GAMnmsetest=nmse(GAMOut,testingtargetmax);
    GAMnrmsetest=nrmse(GAMOut,testingtargetmax);
    GAMmapetest=mape(GAMOut,testingtargetmax);
    GAMr2test=rsquare(GAMOut,testingtargetmax);
    GAMadjr2=adjr2(GAMOut,testingtargetmax,4);
    GAMmpetest=mpe(GAMOut,testingtargetmax);
    GAMsmapetest=smape(GAMOut,testingtargetmax);
    GAMperf50=[GAMmsetest GAMrmsetest GAMmaetest GAMmbetest GAMnmsetest GAMnrmsetest GAMmapetest*100 GAMr2test*100 GAMadjr2*100 GAMmpetest GAMsmapetest];
GAMPtestMax50=GAMPtest50'*maxx;

meantarget50=mean([seriesT(:,1:inputsize)'*maxx; testingtarget50' ]');
meanGAM50=mean([GAMPtestMax50'; GAMOutf50' ]');
figure
plot(meantarget50,height50,'k');
hold on
plot(meanGAM50,height50,'-.g');

hold off
title('average')
legend('measured','GAM est','Location','northwest')

display ('50 m Testing:     MSE     MAPE        MBE        R2')
[GAMperf50]
perfall=[mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 60

nex=2;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y60 = trainingtarget';
Ytest60 = testingtarget';
testingtarget60=Ytest60'*maxx;

testingtargetmax=testingtarget*maxx;
target60=testingtarget60;

%
height60=[height50 60];
mxr=12.5+nex*0.5;
rang60=[0 mxr];
rl60=[1:mxr];
% GAM WSE
%
GAMP60 = [GAMP50; GAMOutf50train'/maxx];
GAMY60 = trainingtarget';
GAMPtest60 = [GAMPtest50; GAMOutf50'/maxx];
GAMYtest60 = testingtarget';
GAMtestingtarget60=GAMYtest60'*maxx;

netGAM = fitrgam(GAMP60',GAMY60);
outval = (predict(netGAM,GAMP60'))';

outvalmax=outval*maxx;
GAMOutf60train=outvalmax';
%mse(GAMOutf60train,GAMY60*maxx)
outvaltest=(predict(netGAM,GAMPtest60'))';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget60;
GAMOut=outvaltestmax;
GAMOutf60=GAMOut;
    GAMmsetest=mse(GAMOut,testingtargetmax);
    GAMrmsetest=rmse(GAMOut,testingtargetmax);
    GAMmaetest=mae(GAMOut,testingtargetmax);
    GAMmbetest=mbe(GAMOut,testingtargetmax);
    GAMnmsetest=nmse(GAMOut,testingtargetmax);
    GAMnrmsetest=nrmse(GAMOut,testingtargetmax);
    GAMmapetest=mape(GAMOut,testingtargetmax);
    GAMr2test=rsquare(GAMOut,testingtargetmax);
    GAMadjr2=adjr2(GAMOut,testingtargetmax,4+(nex-1));
    GAMmpetest=mpe(GAMOut,testingtargetmax);
    GAMsmapetest=smape(GAMOut,testingtargetmax);

    GAMperf60=[GAMmsetest GAMrmsetest GAMmaetest GAMmbetest GAMnmsetest GAMnrmsetest GAMmapetest*100 GAMr2test*100 GAMadjr2*100 GAMmpetest GAMsmapetest];
GAMPtestMax60=GAMPtest60'*maxx;

meantarget60=[meantarget50 mean(testingtarget60)];
meanGAM60=[meanGAM50 mean(GAMOutf60)];

figure
plot(meantarget60,height60,'k');
hold on
plot(meanGAM60,height60,'-.g');

hold off
title('average')
legend('measured','GAM est','Location','northwest')

display ('60 m Testing:     MSE     MAPE        MBE        R2')
[GAMperf60]
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 70

nex=3;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y70 = trainingtarget';
Ytest70 = testingtarget';
testingtarget70=Ytest70'*maxx;

testingtargetmax=testingtarget*maxx;
target70=testingtarget70;

%
height70=[height60 70];
mxr=12.5+nex*0.5;
rang70=[0 mxr];
rl70=[1:mxr];
% GAM WSE
%
GAMP70 = [GAMP60; GAMOutf60train'/maxx];
GAMY70 = trainingtarget';
GAMPtest70 = [GAMPtest60; GAMOutf60'/maxx];
GAMYtest70 = testingtarget';
GAMtestingtarget70=GAMYtest70'*maxx;

netGAM = fitrgam(GAMP70',GAMY70);
outval = (predict(netGAM,GAMP70'))';

outvalmax=outval*maxx;
GAMOutf70train=outvalmax';
%mse(GAMOutf70train,GAMY70*maxx)
outvaltest=(predict(netGAM,GAMPtest70'))';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget70;
GAMOut=outvaltestmax;
GAMOutf70=GAMOut;
    GAMmsetest=mse(GAMOut,testingtargetmax);
    GAMrmsetest=rmse(GAMOut,testingtargetmax);
    GAMmaetest=mae(GAMOut,testingtargetmax);
    GAMmbetest=mbe(GAMOut,testingtargetmax);
    GAMnmsetest=nmse(GAMOut,testingtargetmax);
    GAMnrmsetest=nrmse(GAMOut,testingtargetmax);
    GAMmapetest=mape(GAMOut,testingtargetmax);
    GAMr2test=rsquare(GAMOut,testingtargetmax);
    GAMadjr2=adjr2(GAMOut,testingtargetmax,4+(nex-1));     GAMmpetest=mpe(GAMOut,testingtargetmax);     GAMsmapetest=smape(GAMOut,testingtargetmax); 
    GAMperf70=[GAMmsetest GAMrmsetest GAMmaetest GAMmbetest GAMnmsetest GAMnrmsetest GAMmapetest*100 GAMr2test*100 GAMadjr2*100 GAMmpetest GAMsmapetest];
GAMPtestMax70=GAMPtest70'*maxx;

meantarget70=[meantarget60 mean(testingtarget70)];
meanGAM70=[meanGAM60 mean(GAMOutf70)];

figure
plot(meantarget70,height70,'k');
hold on
plot(meanGAM70,height70,'-.g');

hold off
title('average')
legend('measured','GAM est','Location','northwest')

display ('70 m Testing:     MSE     MAPE        MBE        R2')
[GAMperf70]
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 80

nex=4;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y80 = trainingtarget';
Ytest80 = testingtarget';
testingtarget80=Ytest80'*maxx;

testingtargetmax=testingtarget*maxx;
target80=testingtarget80;

%
height80=[height70 80];
mxr=12.5+nex*0.5;
rang80=[0 mxr];
rl80=[1:mxr];
% GAM WSE
%
GAMP80 = [GAMP70; GAMOutf70train'/maxx];
GAMY80 = trainingtarget';
GAMPtest80 = [GAMPtest70; GAMOutf70'/maxx];
GAMYtest80 = testingtarget';
GAMtestingtarget80=GAMYtest80'*maxx;

netGAM = fitrgam(GAMP80',GAMY80);
outval = (predict(netGAM,GAMP80'))';

outvalmax=outval*maxx;
GAMOutf80train=outvalmax';
%mse(GAMOutf80train,GAMY80*maxx)
outvaltest=(predict(netGAM,GAMPtest80'))';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget80;
GAMOut=outvaltestmax;
GAMOutf80=GAMOut;
    GAMmsetest=mse(GAMOut,testingtargetmax);
    GAMrmsetest=rmse(GAMOut,testingtargetmax);
    GAMmaetest=mae(GAMOut,testingtargetmax);
    GAMmbetest=mbe(GAMOut,testingtargetmax);
    GAMnmsetest=nmse(GAMOut,testingtargetmax);
    GAMnrmsetest=nrmse(GAMOut,testingtargetmax);
    GAMmapetest=mape(GAMOut,testingtargetmax);
    GAMr2test=rsquare(GAMOut,testingtargetmax);
    GAMadjr2=adjr2(GAMOut,testingtargetmax,4+(nex-1));     GAMmpetest=mpe(GAMOut,testingtargetmax);     GAMsmapetest=smape(GAMOut,testingtargetmax); 
    GAMperf80=[GAMmsetest GAMrmsetest GAMmaetest GAMmbetest GAMnmsetest GAMnrmsetest GAMmapetest*100 GAMr2test*100 GAMadjr2*100 GAMmpetest GAMsmapetest];
GAMPtestMax80=GAMPtest80'*maxx;

meantarget80=[meantarget70 mean(testingtarget80)];
meanGAM80=[meanGAM70 mean(GAMOutf80)];

figure
plot(meantarget80,height80,'k');
hold on
plot(meanGAM80,height80,'-.g');

hold off
title('average')
legend('measured','GAM est','Location','northwest')

display ('80 m Testing:     MSE     MAPE        MBE        R2')
[GAMperf80]
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 90

nex=5;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y90 = trainingtarget';
Ytest90 = testingtarget';
testingtarget90=Ytest90'*maxx;

testingtargetmax=testingtarget*maxx;
target90=testingtarget90;

%
height90=[height80 90];
mxr=12.5+nex*0.5;
rang90=[0 mxr];
rl90=[1:mxr];
% GAM WSE
%
GAMP90 = [GAMP80; GAMOutf80train'/maxx];
GAMY90 = trainingtarget';
GAMPtest90 = [GAMPtest80; GAMOutf80'/maxx];
GAMYtest90 = testingtarget';
GAMtestingtarget90=GAMYtest90'*maxx;

netGAM = fitrgam(GAMP90',GAMY90);
outval = (predict(netGAM,GAMP90'))';

outvalmax=outval*maxx;
GAMOutf90train=outvalmax';
%mse(GAMOutf90train,GAMY90*maxx)
outvaltest=(predict(netGAM,GAMPtest90'))';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget90;
GAMOut=outvaltestmax;
GAMOutf90=GAMOut;
    GAMmsetest=mse(GAMOut,testingtargetmax);
    GAMrmsetest=rmse(GAMOut,testingtargetmax);
    GAMmaetest=mae(GAMOut,testingtargetmax);
    GAMmbetest=mbe(GAMOut,testingtargetmax);
    GAMnmsetest=nmse(GAMOut,testingtargetmax);
    GAMnrmsetest=nrmse(GAMOut,testingtargetmax);
    GAMmapetest=mape(GAMOut,testingtargetmax);
    GAMr2test=rsquare(GAMOut,testingtargetmax);
    GAMadjr2=adjr2(GAMOut,testingtargetmax,4+(nex-1));     GAMmpetest=mpe(GAMOut,testingtargetmax);     GAMsmapetest=smape(GAMOut,testingtargetmax); 
    GAMperf90=[GAMmsetest GAMrmsetest GAMmaetest GAMmbetest GAMnmsetest GAMnrmsetest GAMmapetest*100 GAMr2test*100 GAMadjr2*100 GAMmpetest GAMsmapetest];
GAMPtestMax90=GAMPtest90'*maxx;

meantarget90=[meantarget80 mean(testingtarget90)];
meanGAM90=[meanGAM80 mean(GAMOutf90)];

figure
plot(meantarget90,height90,'k');
hold on
plot(meanGAM90,height90,'-.g');

hold off
title('average')
legend('measured','GAM est','Location','northwest')

display ('90 m Testing:     MSE     MAPE        MBE        R2')
[GAMperf90]
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 100

nex=6;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y100 = trainingtarget';
Ytest100 = testingtarget';
testingtarget100=Ytest100'*maxx;

testingtargetmax=testingtarget*maxx;
target100=testingtarget100;

%
height100=[height90 100];
mxr=12.5+nex*0.5;
rang100=[0 mxr];
rl100=[1:mxr];
% GAM WSE
%
GAMP100 = [GAMP90; GAMOutf90train'/maxx];
GAMY100 = trainingtarget';
GAMPtest100 = [GAMPtest90; GAMOutf90'/maxx];
GAMYtest100 = testingtarget';
GAMtestingtarget100=GAMYtest100'*maxx;

netGAM = fitrgam(GAMP100',GAMY100);
outval = (predict(netGAM,GAMP100'))';

outvalmax=outval*maxx;
GAMOutf100train=outvalmax';
%mse(GAMOutf100train,GAMY100*maxx)
outvaltest=(predict(netGAM,GAMPtest100'))';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget100;
GAMOut=outvaltestmax;
GAMOutf100=GAMOut;
    GAMmsetest=mse(GAMOut,testingtargetmax);
    GAMrmsetest=rmse(GAMOut,testingtargetmax);
    GAMmaetest=mae(GAMOut,testingtargetmax);
    GAMmbetest=mbe(GAMOut,testingtargetmax);
    GAMnmsetest=nmse(GAMOut,testingtargetmax);
    GAMnrmsetest=nrmse(GAMOut,testingtargetmax);
    GAMmapetest=mape(GAMOut,testingtargetmax);
    GAMr2test=rsquare(GAMOut,testingtargetmax);
    GAMadjr2=adjr2(GAMOut,testingtargetmax,4+(nex-1));     GAMmpetest=mpe(GAMOut,testingtargetmax);     GAMsmapetest=smape(GAMOut,testingtargetmax); 
    GAMperf100=[GAMmsetest GAMrmsetest GAMmaetest GAMmbetest GAMnmsetest GAMnrmsetest GAMmapetest*100 GAMr2test*100 GAMadjr2*100 GAMmpetest GAMsmapetest];
GAMPtestMax100=GAMPtest100'*maxx;

meantarget100=[meantarget90 mean(testingtarget100)];
meanGAM100=[meanGAM90 mean(GAMOutf100)];

figure
plot(meantarget100,height100,'k');
hold on
plot(meanGAM100,height100,'-.g');

hold off
title('average')
legend('measured','GAM est','Location','northwest')

display ('100 m Testing:     MSE     MAPE        MBE        R2')
[GAMperf100]
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 110

nex=7;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y110 = trainingtarget';
Ytest110 = testingtarget';
testingtarget110=Ytest110'*maxx;

testingtargetmax=testingtarget*maxx;
target110=testingtarget110;

%
height110=[height100 110];
mxr=12.5+nex*0.5;
rang110=[0 mxr];
rl110=[1:mxr];
% GAM WSE
%
GAMP110 = [GAMP100; GAMOutf100train'/maxx];
GAMY110 = trainingtarget';
GAMPtest110 = [GAMPtest100; GAMOutf100'/maxx];
GAMYtest110 = testingtarget';
GAMtestingtarget110=GAMYtest110'*maxx;

netGAM = fitrgam(GAMP110',GAMY110);
outval = (predict(netGAM,GAMP110'))';

outvalmax=outval*maxx;
GAMOutf110train=outvalmax';
%mse(GAMOutf110train,GAMY110*maxx)
outvaltest=(predict(netGAM,GAMPtest110'))';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget110;
GAMOut=outvaltestmax;
GAMOutf110=GAMOut;
    GAMmsetest=mse(GAMOut,testingtargetmax);
    GAMrmsetest=rmse(GAMOut,testingtargetmax);
    GAMmaetest=mae(GAMOut,testingtargetmax);
    GAMmbetest=mbe(GAMOut,testingtargetmax);
    GAMnmsetest=nmse(GAMOut,testingtargetmax);
    GAMnrmsetest=nrmse(GAMOut,testingtargetmax);
    GAMmapetest=mape(GAMOut,testingtargetmax);
    GAMr2test=rsquare(GAMOut,testingtargetmax);
    GAMadjr2=adjr2(GAMOut,testingtargetmax,4+(nex-1));     GAMmpetest=mpe(GAMOut,testingtargetmax);     GAMsmapetest=smape(GAMOut,testingtargetmax); 
    GAMperf110=[GAMmsetest GAMrmsetest GAMmaetest GAMmbetest GAMnmsetest GAMnrmsetest GAMmapetest*100 GAMr2test*100 GAMadjr2*100 GAMmpetest GAMsmapetest];
GAMPtestMax110=GAMPtest110'*maxx;

meantarget110=[meantarget100 mean(testingtarget110)];
meanGAM110=[meanGAM100 mean(GAMOutf110)];

figure
plot(meantarget110,height110,'k');
hold on
plot(meanGAM110,height110,'-.g');

hold off
title('average')
legend('measured','GAM est','Location','northwest')

display ('110 m Testing:     MSE     MAPE        MBE        R2')
[GAMperf110]
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 120

nex=8;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y120 = trainingtarget';
Ytest120 = testingtarget';
testingtarget120=Ytest120'*maxx;

testingtargetmax=testingtarget*maxx;
target120=testingtarget120;

%
height120=[height110 120];
mxr=12.5+nex*0.5;
rang120=[0 mxr];
rl120=[1:mxr];
% GAM WSE
%
GAMP120 = [GAMP110; GAMOutf110train'/maxx];
GAMY120 = trainingtarget';
GAMPtest120 = [GAMPtest110; GAMOutf110'/maxx];
GAMYtest120 = testingtarget';
GAMtestingtarget120=GAMYtest120'*maxx;

netGAM = fitrgam(GAMP120',GAMY120);
outval = (predict(netGAM,GAMP120'))';

outvalmax=outval*maxx;
GAMOutf120train=outvalmax';
%mse(GAMOutf120train,GAMY120*maxx)
outvaltest=(predict(netGAM,GAMPtest120'))';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget120;
GAMOut=outvaltestmax;
GAMOutf120=GAMOut;
    GAMmsetest=mse(GAMOut,testingtargetmax);
    GAMrmsetest=rmse(GAMOut,testingtargetmax);
    GAMmaetest=mae(GAMOut,testingtargetmax);
    GAMmbetest=mbe(GAMOut,testingtargetmax);
    GAMnmsetest=nmse(GAMOut,testingtargetmax);
    GAMnrmsetest=nrmse(GAMOut,testingtargetmax);
    GAMmapetest=mape(GAMOut,testingtargetmax);
    GAMr2test=rsquare(GAMOut,testingtargetmax);
    GAMadjr2=adjr2(GAMOut,testingtargetmax,4+(nex-1));     GAMmpetest=mpe(GAMOut,testingtargetmax);     GAMsmapetest=smape(GAMOut,testingtargetmax); 
    GAMperf120=[GAMmsetest GAMrmsetest GAMmaetest GAMmbetest GAMnmsetest GAMnrmsetest GAMmapetest*100 GAMr2test*100 GAMadjr2*100 GAMmpetest GAMsmapetest];
GAMPtestMax120=GAMPtest120'*maxx;

meantarget120=[meantarget110 mean(testingtarget120)];
meanGAM120=[meanGAM110 mean(GAMOutf120)];

figure
plot(meantarget120,height120,'k');
hold on
plot(meanGAM120,height120,'-.g');

hold off
title('average')
legend('measured','GAM est','Location','northwest')

display ('120 m Testing:     MSE     MAPE        MBE        R2')
[GAMperf120]
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%%
%% 130

nex=9;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y130 = trainingtarget';
Ytest130 = testingtarget';
testingtarget130=Ytest130'*maxx;

testingtargetmax=testingtarget*maxx;
target130=testingtarget130;

%
height130=[height120 130];
mxr=12.5+nex*0.5;
rang130=[0 mxr];
rl130=[1:mxr];
% GAM WSE
%
GAMP130 = [GAMP120; GAMOutf120train'/maxx];
GAMY130 = trainingtarget';
GAMPtest130 = [GAMPtest120; GAMOutf120'/maxx];
GAMYtest130 = testingtarget';
GAMtestingtarget130=GAMYtest130'*maxx;

netGAM = fitrgam(GAMP130',GAMY130);
outval = (predict(netGAM,GAMP130'))';

outvalmax=outval*maxx;
GAMOutf130train=outvalmax';
%mse(GAMOutf130train,GAMY130*maxx)
outvaltest=(predict(netGAM,GAMPtest130'))';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget130;
GAMOut=outvaltestmax;
GAMOutf130=GAMOut;
    GAMmsetest=mse(GAMOut,testingtargetmax);
    GAMrmsetest=rmse(GAMOut,testingtargetmax);
    GAMmaetest=mae(GAMOut,testingtargetmax);
    GAMmbetest=mbe(GAMOut,testingtargetmax);
    GAMnmsetest=nmse(GAMOut,testingtargetmax);
    GAMnrmsetest=nrmse(GAMOut,testingtargetmax);
    GAMmapetest=mape(GAMOut,testingtargetmax);
    GAMr2test=rsquare(GAMOut,testingtargetmax);
    GAMadjr2=adjr2(GAMOut,testingtargetmax,4+(nex-1));     GAMmpetest=mpe(GAMOut,testingtargetmax);     GAMsmapetest=smape(GAMOut,testingtargetmax); 
    GAMperf130=[GAMmsetest GAMrmsetest GAMmaetest GAMmbetest GAMnmsetest GAMnrmsetest GAMmapetest*100 GAMr2test*100 GAMadjr2*100 GAMmpetest GAMsmapetest];
GAMPtestMax130=GAMPtest130'*maxx;

meantarget130=[meantarget120 mean(testingtarget130)];
meanGAM130=[meanGAM120 mean(GAMOutf130)];

figure
plot(meantarget130,height130,'k');
hold on
plot(meanGAM130,height130,'-.g');

hold off
title('average')
legend('measured','GAM est','Location','northwest')

display ('130 m Testing:     MSE     MAPE        MBE        R2')
[GAMperf130]
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 140

nex=10;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y140 = trainingtarget';
Ytest140 = testingtarget';
testingtarget140=Ytest140'*maxx;

testingtargetmax=testingtarget*maxx;
target140=testingtarget140;

%
height140=[height130 140];
mxr=12.5+nex*0.5;
rang140=[0 mxr];
rl140=[1:mxr];
% GAM WSE
%
GAMP140 = [GAMP130; GAMOutf130train'/maxx];
GAMY140 = trainingtarget';
GAMPtest140 = [GAMPtest130; GAMOutf130'/maxx];
GAMYtest140 = testingtarget';
GAMtestingtarget140=GAMYtest140'*maxx;

netGAM = fitrgam(GAMP140',GAMY140);
outval = (predict(netGAM,GAMP140'))';

outvalmax=outval*maxx;
GAMOutf140train=outvalmax';
%mse(GAMOutf140train,GAMY140*maxx)
outvaltest=(predict(netGAM,GAMPtest140'))';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget140;
GAMOut=outvaltestmax;
GAMOutf140=GAMOut;
    GAMmsetest=mse(GAMOut,testingtargetmax);
    GAMrmsetest=rmse(GAMOut,testingtargetmax);
    GAMmaetest=mae(GAMOut,testingtargetmax);
    GAMmbetest=mbe(GAMOut,testingtargetmax);
    GAMnmsetest=nmse(GAMOut,testingtargetmax);
    GAMnrmsetest=nrmse(GAMOut,testingtargetmax);
    GAMmapetest=mape(GAMOut,testingtargetmax);
    GAMr2test=rsquare(GAMOut,testingtargetmax);
    GAMadjr2=adjr2(GAMOut,testingtargetmax,4+(nex-1));     GAMmpetest=mpe(GAMOut,testingtargetmax);     GAMsmapetest=smape(GAMOut,testingtargetmax); 
    GAMperf140=[GAMmsetest GAMrmsetest GAMmaetest GAMmbetest GAMnmsetest GAMnrmsetest GAMmapetest*100 GAMr2test*100 GAMadjr2*100 GAMmpetest GAMsmapetest];
GAMPtestMax140=GAMPtest140'*maxx;

meantarget140=[meantarget130 mean(testingtarget140)];
meanGAM140=[meanGAM130 mean(GAMOutf140)];

figure
plot(meantarget140,height140,'k');
hold on
plot(meanGAM140,height140,'-.g');

hold off
title('average')
legend('measured','GAM est','Location','northwest')

display ('140 m Testing:     MSE     MAPE        MBE        R2')
[GAMperf140]
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];


%% 150

nex=11;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y150 = trainingtarget';
Ytest150 = testingtarget';
testingtarget150=Ytest150'*maxx;

testingtargetmax=testingtarget*maxx;
target150=testingtarget150;

%
height150=[height140 150];
mxr=12.5+nex*0.5;
rang150=[0 mxr];
rl150=[1:mxr];
% GAM WSE
%
GAMP150 = [GAMP140; GAMOutf140train'/maxx];
GAMY150 = trainingtarget';
GAMPtest150 = [GAMPtest140; GAMOutf140'/maxx];
GAMYtest150 = testingtarget';
GAMtestingtarget150=GAMYtest150'*maxx;

netGAM = fitrgam(GAMP150',GAMY150);
outval = (predict(netGAM,GAMP150'))';

outvalmax=outval*maxx;
GAMOutf150train=outvalmax';
%mse(GAMOutf150train,GAMY150*maxx)
outvaltest=(predict(netGAM,GAMPtest150'))';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget150;
GAMOut=outvaltestmax;
GAMOutf150=GAMOut;
    GAMmsetest=mse(GAMOut,testingtargetmax);
    GAMrmsetest=rmse(GAMOut,testingtargetmax);
    GAMmaetest=mae(GAMOut,testingtargetmax);
    GAMmbetest=mbe(GAMOut,testingtargetmax);
    GAMnmsetest=nmse(GAMOut,testingtargetmax);
    GAMnrmsetest=nrmse(GAMOut,testingtargetmax);
    GAMmapetest=mape(GAMOut,testingtargetmax);
    GAMr2test=rsquare(GAMOut,testingtargetmax);
    GAMadjr2=adjr2(GAMOut,testingtargetmax,4+(nex-1));     GAMmpetest=mpe(GAMOut,testingtargetmax);     GAMsmapetest=smape(GAMOut,testingtargetmax); 
    GAMperf150=[GAMmsetest GAMrmsetest GAMmaetest GAMmbetest GAMnmsetest GAMnrmsetest GAMmapetest*100 GAMr2test*100 GAMadjr2*100 GAMmpetest GAMsmapetest];
GAMPtestMax150=GAMPtest150'*maxx;

meantarget150=[meantarget140 mean(testingtarget150)];
meanGAM150=[meanGAM140 mean(GAMOutf150)];

figure
plot(meantarget150,height150,'k');
hold on
plot(meanGAM150,height150,'-.g');

hold off
title('average')
legend('measured','GAM est','Location','northwest')

display ('150 m Testing:     MSE     MAPE        MBE        R2')
[GAMperf150]
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 160

nex=12;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y160 = trainingtarget';
Ytest160 = testingtarget';
testingtarget160=Ytest160'*maxx;

testingtargetmax=testingtarget*maxx;
target160=testingtarget160;

%
height160=[height150 160];
mxr=12.5+nex*0.5;
rang160=[0 mxr];
rl160=[1:mxr];
% GAM WSE
%
GAMP160 = [GAMP150; GAMOutf150train'/maxx];
GAMY160 = trainingtarget';
GAMPtest160 = [GAMPtest150; GAMOutf150'/maxx];
GAMYtest160 = testingtarget';
GAMtestingtarget160=GAMYtest160'*maxx;

netGAM = fitrgam(GAMP160',GAMY160);
outval = (predict(netGAM,GAMP160'))';

outvalmax=outval*maxx;
GAMOutf160train=outvalmax';
%mse(GAMOutf160train,GAMY160*maxx)
outvaltest=(predict(netGAM,GAMPtest160'))';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget160;
GAMOut=outvaltestmax;
GAMOutf160=GAMOut;
    GAMmsetest=mse(GAMOut,testingtargetmax);
    GAMrmsetest=rmse(GAMOut,testingtargetmax);
    GAMmaetest=mae(GAMOut,testingtargetmax);
    GAMmbetest=mbe(GAMOut,testingtargetmax);
    GAMnmsetest=nmse(GAMOut,testingtargetmax);
    GAMnrmsetest=nrmse(GAMOut,testingtargetmax);
    GAMmapetest=mape(GAMOut,testingtargetmax);
    GAMr2test=rsquare(GAMOut,testingtargetmax);
    GAMadjr2=adjr2(GAMOut,testingtargetmax,4+(nex-1));     GAMmpetest=mpe(GAMOut,testingtargetmax);     GAMsmapetest=smape(GAMOut,testingtargetmax); 
    GAMperf160=[GAMmsetest GAMrmsetest GAMmaetest GAMmbetest GAMnmsetest GAMnrmsetest GAMmapetest*100 GAMr2test*100 GAMadjr2*100 GAMmpetest GAMsmapetest];
GAMPtestMax160=GAMPtest160'*maxx;

meantarget160=[meantarget150 mean(testingtarget160)];
meanGAM160=[meanGAM150 mean(GAMOutf160)];

figure
plot(meantarget160,height160,'k');
hold on
plot(meanGAM160,height160,'-.g');

hold off
title('average')
legend('measured','GAM est','Location','northwest')

display ('160 m Testing:     MSE     MAPE        MBE        R2')
[GAMperf160]
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];
%% 170

nex=13;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y170 = trainingtarget';
Ytest170 = testingtarget';
testingtarget170=Ytest170'*maxx;

testingtargetmax=testingtarget*maxx;
target170=testingtarget170;

%
height170=[height160 170];
mxr=12.5+nex*0.5;
rang170=[0 mxr];
rl170=[1:mxr];
% GAM WSE
%
GAMP170 = [GAMP160; GAMOutf160train'/maxx];
GAMY170 = trainingtarget';
GAMPtest170 = [GAMPtest160; GAMOutf160'/maxx];
GAMYtest170 = testingtarget';
GAMtestingtarget170=GAMYtest170'*maxx;

netGAM = fitrgam(GAMP170',GAMY170);
outval = (predict(netGAM,GAMP170'))';

outvalmax=outval*maxx;
GAMOutf170train=outvalmax';
%mse(GAMOutf170train,GAMY170*maxx)
outvaltest=(predict(netGAM,GAMPtest170'))';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget170;
GAMOut=outvaltestmax;
GAMOutf170=GAMOut;
    GAMmsetest=mse(GAMOut,testingtargetmax);
    GAMrmsetest=rmse(GAMOut,testingtargetmax);
    GAMmaetest=mae(GAMOut,testingtargetmax);
    GAMmbetest=mbe(GAMOut,testingtargetmax);
    GAMnmsetest=nmse(GAMOut,testingtargetmax);
    GAMnrmsetest=nrmse(GAMOut,testingtargetmax);
    GAMmapetest=mape(GAMOut,testingtargetmax);
    GAMr2test=rsquare(GAMOut,testingtargetmax);
    GAMadjr2=adjr2(GAMOut,testingtargetmax,4+(nex-1));     GAMmpetest=mpe(GAMOut,testingtargetmax);     GAMsmapetest=smape(GAMOut,testingtargetmax); 
    GAMperf170=[GAMmsetest GAMrmsetest GAMmaetest GAMmbetest GAMnmsetest GAMnrmsetest GAMmapetest*100 GAMr2test*100 GAMadjr2*100 GAMmpetest GAMsmapetest];
GAMPtestMax170=GAMPtest170'*maxx;

meantarget170=[meantarget160 mean(testingtarget170)];
meanGAM170=[meanGAM160 mean(GAMOutf170)];

figure
plot(meantarget170,height170,'k');
hold on
plot(meanGAM170,height170,'-.g');

hold off
title('average')
legend('measured','GAM est','Location','northwest')

display ('170 m Testing:     MSE     MAPE        MBE        R2')
[GAMperf170]
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 180

nex=14;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y180 = trainingtarget';
Ytest180 = testingtarget';
testingtarget180=Ytest180'*maxx;

testingtargetmax=testingtarget*maxx;
target180=testingtarget180;

%
height180=[height170 180];
mxr=12.5+nex*0.5;
rang180=[0 mxr];
rl180=[1:mxr];
% GAM WSE
%
GAMP180 = [GAMP170; GAMOutf170train'/maxx];
GAMY180 = trainingtarget';
GAMPtest180 = [GAMPtest170; GAMOutf170'/maxx];
GAMYtest180 = testingtarget';
GAMtestingtarget180=GAMYtest180'*maxx;

netGAM = fitrgam(GAMP180',GAMY180);
outval = (predict(netGAM,GAMP180'))';

outvalmax=outval*maxx;
GAMOutf180train=outvalmax';
%mse(GAMOutf180train,GAMY180*maxx)
outvaltest=(predict(netGAM,GAMPtest180'))';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget180;
GAMOut=outvaltestmax;
GAMOutf180=GAMOut;
    GAMmsetest=mse(GAMOut,testingtargetmax);
    GAMrmsetest=rmse(GAMOut,testingtargetmax);
    GAMmaetest=mae(GAMOut,testingtargetmax);
    GAMmbetest=mbe(GAMOut,testingtargetmax);
    GAMnmsetest=nmse(GAMOut,testingtargetmax);
    GAMnrmsetest=nrmse(GAMOut,testingtargetmax);
    GAMmapetest=mape(GAMOut,testingtargetmax);
    GAMr2test=rsquare(GAMOut,testingtargetmax);
    GAMadjr2=adjr2(GAMOut,testingtargetmax,4+(nex-1));     GAMmpetest=mpe(GAMOut,testingtargetmax);     GAMsmapetest=smape(GAMOut,testingtargetmax); 
    GAMperf180=[GAMmsetest GAMrmsetest GAMmaetest GAMmbetest GAMnmsetest GAMnrmsetest GAMmapetest*100 GAMr2test*100 GAMadjr2*100 GAMmpetest GAMsmapetest];
GAMPtestMax180=GAMPtest180'*maxx;

meantarget180=[meantarget170 mean(testingtarget180)];
meanGAM180=[meanGAM170 mean(GAMOutf180)];

figure
plot(meantarget180,height180,'k');
hold on
plot(meanGAM180,height180,'-.g');

hold off
title('average')
legend('measured','GAM est','Location','northwest')

display ('180 m Testing:     MSE     MAPE        MBE        R2')
[GAMperf180]
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];


%%
GAMPerfAll=[GAMperf50;GAMperf60;GAMperf70;GAMperf80;GAMperf90;GAMperf100;GAMperf110;GAMperf120; GAMperf130; GAMperf140; GAMperf150; GAMperf160; GAMperf170; ; GAMperf180];

