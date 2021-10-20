% ================================================= %
% Implements the geometry-aware PCA on GBW geometry 
% ================================================= %
clear;
clc;
rng('default');
rng(0);

data_name = 'ETH_data';

% ETH
load(data_name);
X = ETH_data;

d = size(X, 1);
rs = [5 10 30 50 70 90];

N = size(X, 3);
k = 1;


Y_bw_s = {};
for i_r = 1:length(rs)
    r = rs(i_r);
    
    % init
    Y0 = qr_unique(randn(d, r));
    
    args.init = Y0;
    args.maxepoch = 20;
    args.tolgradnorm = 1e-10;

    [Y_bw, info] = BWPCA(X, r, args);    
    
    Y_bw_s{i_r} = Y_bw;
end



%% BWKNN 
repeat = 10;
r_list = [0 5 10 30 50 70 90];
for i_rep = 1:repeat

    train_ratio = 0.5;
    
    % train-test split
    [X_train, X_test, y_train, y_test] = train_test_split(X, labels, train_ratio);    
    
    le = [];
    bw = [];
    ai = [];
    for i_r = 1:length(r_list)
        r = r_list(i_r);
        if r == 0
            bw_res = KNNClassifier(X_train, y_train, X_test, y_test,k,eye(d),'BW');
        else
            Y_bw = Y_bw_s{i_r -1};
            
            X_train_bw = reduce_dim(X_train, Y_bw);
            X_test_bw = reduce_dim(X_test, Y_bw);
                        
            bw_res = KNNClassifier(X_train_bw, y_train, X_test_bw, y_test,k,eye(r),'BW');
        end
        
        bw = [bw bw_res];

    end
    final_result{i_rep}.bw = bw;
        
end

% compute the average accuracy
[mean_acc, std_acc] = acc_mean(final_result, r_list);
