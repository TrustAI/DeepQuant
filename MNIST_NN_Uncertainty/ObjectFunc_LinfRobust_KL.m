function value = ObjectFunc_LinfRobust_KL(x)
% load temp
global layer
global convnet
global fVal_x0
global fInd_x0
global x0Vec
global delta
global resultCell
global cell_ind
% global lipConst
global value_Lip
global fAll

% global lipConst
% cell_ind = cell_ind + 1;

rowSize = length(x)^0.5;
X_image = reshape(x,[rowSize,rowSize,1]);
fVal_all = activations(convnet,X_image,layer,'OutputAs','rows');
% [~,fVal_all_ind] = max(fVal_all);
fVal_j = fVal_all(:,fInd_x0);
m = length(fAll);
s_kl_x0 = - sum(1/m*log(m*fAll));
s_kl_x1 = - sum(1/m*log10(m*fVal_all));
w_denominator = abs(s_kl_x0 - s_kl_x1);
% value = w_denominator;
% w_numerator = max(abs((x - x0Vec)));
%  if w_denominator==0
%      value = 1000;
%  else
%  value = w_denominator;
%  end

value = s_kl_x1;


