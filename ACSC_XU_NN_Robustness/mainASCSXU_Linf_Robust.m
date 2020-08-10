
clear;
clc;

%% set up global variables
global layer
global convnet
global fVal_x0
global fInd_x0
global x0Vec
global delta
global lipConst
% global resultCell
% global cell_ind

%% load data
load ASCSXuNN
load ASCSXuNN_Data
% convnet = convnet;
layer = 'fc_7';
resultCell = {};
cell_ind = 1;
% for 
    imgInd = 1200;%:20 %% index of image tested
% for 
%     boxConSize = %0.05:0.05:0.7 %% the size of box contrait    
tStart = tic;
epsilonf = 0.001;
pixelRange = [0,1];
delta = 0.00039*0.001;
x0 = TrainData(:,:,:,imgInd);
x0Vec = x0(:);

% boxConSize = 0.3; %% the size of box contrait
fAll = activations(convnet,x0,layer);
rowSize = size(x0,1);

[fVal_x0,fInd_x0] = max(fAll);


boxBound = [100,10000;
            0,6.2832;
            0,6.2832;
            30,300;
            30,300];

x = x0Vec;


%% Stage one

options = optimoptions('patternsearch','Display','iter');
options.MeshContractionFactor = 0.8;
options.MeshExpansionFactor = 1.3;
options.MeshTolerance = 0.00001;
options.UseParallel = true;
options.MaxIterations = 1500;
tic
[x_opt,fval_opt,exitflag,output] = patternsearch(@ObjectFuncASCSXU_LinfRobust,x,[],[],[],[],...
    boxBound(:,1),boxBound(:,2),options);
toc
%%

resultCell{cell_ind,1} = x0;
resultCell{cell_ind,2} = reshape(x_opt,[rowSize,1,1]);
resultCell{cell_ind,3} = 1/fval_opt
% resultCell{cell_ind,4} = boxConSize;
resultCell{cell_ind,4} = imgInd;

cell_ind = cell_ind + 1;
% end
% end
% save MNIST_resultAll_1_20d resultCell

