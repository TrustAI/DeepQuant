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
global resultCell
global cell_ind
global fAll
%% prepare data
load DNN7
convnet = convnet1;
%layer = 'fc_2';
layer = 'softmax';
allResult = {};
resultCell = {};
cell_ind = 1;
for  imgInd = [6,10,13,17,19,25] %1:20 %% index of image tested

 for   boxConSize = 0.05:0.05:0.5 %% the size of box contrait

%     imgInd = 6; %1:20 %% index of image tested
tStart = tic;
epsilonf = 0.001;
pixelRange = [0,1];
delta = 0.0039*0.001;
x0 = XTest(:,:,:,imgInd);
x0Vec = x0(:);
% boxConSize = 0.5; %% the size of box contrait

fAll = activations(convnet,x0,layer,'OutputAs','rows');
rowSize = size(x0,1);

[fVal_x0,fInd_x0] = max(fAll);

boxBound = BoxBoundsConst(x0Vec,boxConSize,pixelRange);

x = x0Vec;
initalKL = ObjectFunc_LinfLocalRobust_KL(x);
resultCell{cell_ind,1} = x0;
resultCell{cell_ind,2} = 0;
resultCell{cell_ind,3} = fInd_x0;
resultCell{cell_ind,4} = 0;

%% Stage one

%         options = optimoptions('patternsearch','Display','off',...
%             'OutputFcn',@StopFunc);
options = optimoptions('patternsearch','Display','iter');
options.MeshContractionFactor = 0.5;
options.MeshExpansionFactor = 2;
options.MeshTolerance = 0.0039/2;
options.UseParallel = true;
options.MaxIterations = 2000;
[x_opt,fval_opt,exitflag,output] = patternsearch(@ObjectFunc_LinfRobust_KL,x,[],[],[],[],...
    boxBound(:,1),boxBound(:,2),options);

%%
disp(fAll)
fVal_all = activations(convnet,reshape(x_opt,[rowSize,rowSize]),layer,'OutputAs','rows');
disp(fVal_all)

%%
% figure;

subplot(2,2,1);
imshow(x0);
title(['True Image, KL = ' num2str(initalKL,3)])
subplot(2,2,2);
bar(fAll);
ylabel('Probability')
xlabel('Classfication Category')
%  axis([inf inf 0 1])
 grid on
subplot(2,2,3);
imshow(reshape(x_opt,[rowSize,rowSize]));
title(['Uncertainty Image, KL =' num2str(fval_opt,3) ])
subplot(2,2,4);
bar(fVal_all)
ylabel('Probability')
xlabel('Classfication Category')
grid on

saveas(gcf, ['MNIST_KLnew_' num2str(imgInd) '_d' num2str(boxConSize,1) '_Iter'...
    num2str(options.MaxIterations) '_KL' num2str(fval_opt,3) '.fig']);
saveas(gcf, ['MNIST_KLnew_' num2str(imgInd) '_d' num2str(boxConSize,1) '_Iter'...
    num2str(options.MaxIterations) '_KL' num2str(fval_opt,3) '.png']);

resultCell{cell_ind,1} = x0;
resultCell{cell_ind,2} = reshape(x_opt,[rowSize,rowSize,1]);
resultCell{cell_ind,3} = 1/fval_opt;
resultCell{cell_ind,4} = boxConSize;
resultCell{cell_ind,5} = fval_opt;
resultCell{cell_ind,6} = imgInd;
resultCell{cell_ind,7} = initalKL;

cell_ind = cell_ind + 1;
save KLnew_mnist_global resultCell
 end
end


