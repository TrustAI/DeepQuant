function value = obj_max(x)

% global layer
global convnet
% global fVal_x0
% global fInd_x0
% global x0Vec
% global delta
% global resultCell
% global cell_ind
% global lipConst
% global value_Lip

% global lipConst
value = -predict(convnet,x);
    
% rowSize = length(x);
% X_image = reshape(x,[rowSize,1,1]);
% fVal_all = activations(convnet,X_image,layer,'OutputAs','rows');
% % [~,fVal_all_ind] = max(fVal_all);
% fVal_j = fVal_all(:,fInd_x0);
% fVal_other = fVal_all;
% fVal_other(:,fInd_x0) = [];
% w_denominator = abs(fVal_x0 - max(fVal_other) - (fVal_j - max(fVal_other)));
% w_numerator = max(abs((x - x0Vec)));
% if w_numerator==0
%     value = 1000;
% else
% value = w_numerator/w_denominator;
% end
% 
% resultCell{cell_ind,1} = X_image;
% resultCell{cell_ind,2} = double(value_Lip);
% resultCell{cell_ind,3} = fVal_all_ind;
% resultCell{cell_ind,4} = -value;

