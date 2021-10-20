function X_reduced = reduce_dim(X,W)
% reduce dimensionality as W'*X*W
%d = size(W, 1);
r = size(W, 2);
N = size(X,3);
X_reduced = zeros(r,r,N);
for ii = 1:N
    X_reduced(:,:,ii) = W'*X(:,:,ii)*W;
end

end

