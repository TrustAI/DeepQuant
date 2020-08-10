%% get the box contraint

function bounds = BoxBoundsConst(X0,d_bound,pixelRange)

X0_min = X0 - d_bound;
X0_max = X0 + d_bound;
pixelRange = repmat(pixelRange,size(X0));

bounds  = [max([pixelRange(:,1),X0_min],[],2),min([pixelRange(:,2),X0_max],[],2)];

end

