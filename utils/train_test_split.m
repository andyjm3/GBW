function [X_train,X_test, y_train, y_test] = train_test_split(X,labels,train_ratio)

% randoly permute data
[X, idx] = shuffle_data(X);
labels = labels(idx);


C = length(unique(labels));
X_train = [];
y_train = [];
X_test = [];
y_test = [];
for cc = 1:C
    X_cc = X(:,:,labels == cc);
    N_cc = size(X_cc, 3);
    
    N_train_cc = floor(N_cc * train_ratio);
    N_test_cc = N_cc - N_train_cc;
    
    X_train_cc = X_cc(:,:,1:N_train_cc);
    X_test_cc = X_cc(:,:,N_train_cc+1:end);
    
    
    X_train = cat(3, X_train, X_train_cc);
    X_test = cat(3, X_test, X_test_cc);
    y_train = [y_train ones(1,N_train_cc)*cc];
    y_test = [y_test ones(1,N_test_cc)*cc];

end



end

