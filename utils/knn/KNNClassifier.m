function [Result] = KNNClassifier(X_train,Y_train, X_test, Y_test, k, M, dist)
%KNNCLASSIFIER_MN Calculates the prediction of knn and its accuracy for
%MNIST dataset
%   Input: X_train, Y_train, X_test, Y_test are standard 3D array
%        : k is number of neighbours; dist represents the distance metric
%   Output: [Accuracy, prediction y of test set]

Dmap = DistMap(X_train,X_test,M,dist);

Ntest = size(Dmap, 2);
pred_y = zeros(1,Ntest);

for i=1:Ntest
    try
        [~,idx] = mink(Dmap(:,i), k);
    catch
        idx = min_k(Dmap(:,i), k);
    end
    pred_y(i) = mode(Y_train(idx));
end

%acc = 1 - nnz(pred_y - Y_test)/Ntest;
[~,Result,~]= confusion.getMatrix(Y_test,pred_y,0);
end

