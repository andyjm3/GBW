function [] = test_gmm()

% Test GMM model between AI, BW metric with benchmark to be EM
% Loss: - sum_{i=1}^{n} log ( sum_{j=1}^{K} p_k q_k(x_i) ) where p_k is a
% in a probability simplex. 
    
    % All saved figures for main text use rng(0).
    rng('default');
    rng(0);
    
    
    data_choice = 'iris';
    solver_choice = 'SGD';
    
    %% data
    
    symm = @(Z) 0.5*(Z+Z');
    
    switch data_choice
            
        case 'iris'
            data_org = iris_dataset;
            K = 3; 
            
        case 'kmeans'
            X = [];
            load kmeansdata;
            data_org = X';
            K = 3;       
        
    end
    [d, N] = size(data_org);
    
    data = [data_org; ones(1,N)];
    
    
    
    %% Problem define
    % define mfd
    powerSPDai = powermanifold(sympositivedefinitefactory(d+1), K);
    powerSPDbw = powermanifold(sympositivedefiniteBWfactory(d+1), K);
    simplex = multinomialfactory(K,1);
    M_ai = productmanifold(struct('D',powerSPDai ,'p', simplex));
    M_bw = productmanifold(struct('D',powerSPDbw ,'p', simplex));
    
    powerSPDle = powermanifold(symmetricfactory(d+1), K);
    M_le = productmanifold(struct('D',powerSPDle ,'p', simplex));
    
    % ========== problem structure for AI metric ============
    problemAI.M = M_ai;
    problemAI.cost = @costAI;
    problemAI.egrad = @egradAI;
    problemAI.partialegrad = @partialegradAI;
    problemAI.ncostterms = N;

    const = (2 * pi)^(-d/2) * exp(1/2);
    
    function f = costAI(theta)        
        f = 0;        
        for k = 1:K
            [~, R, Rinv] = invmat(theta.D{k});
            % diag(X' C^{-1} X)
            u = sum((Rinv' * data).^2, 1); 
            % first take log then exp is faster than calcualte det
            f = f + exp(log(theta.p(k)) - sum(log(diag(R))) - 0.5 * u);
        end 
        f = - 1/N * sum(log(const * f));
    end

    function g = egradAI(theta)
        g.D = cell(K,1);
        lik = zeros(K, N); % a K-by-N matrix with p_k q_k(x_i)
        Sigma_inv = cell(K,1);
        for k = 1:K
            [Sinv, R, Rinv] = invmat(theta.D{k});
            u = sum((Rinv' * data).^2, 1); 
            temp = const * exp(log(theta.p(k)) - sum(log(diag(R))) - 0.5 * u);            
            lik(k, :) = temp;
            Sigma_inv{k} = Sinv;
        end
        component_weights = lik ./ sum(lik, 1);
         
        for k = 1:K
            % Multiply data by the square root of the weight, follows from
            % mvn2factory line 175 - 179
            weight = component_weights(k,:);
            sumW = sum(weight, 2);
            weight = sqrt(weight);            
            data_k = bsxfun(@times, data, weight);
            DDT = (data_k * data_k.');
            Sinv = Sigma_inv{k};
            
            g.D{k} = - 1/(2*N) * ( -sumW * Sinv + Sinv * DDT *Sinv);
        end
                
        g.p = -1/N *squeeze(sum(component_weights, 2)./theta.p);
    end

    function g = partialegradAI(theta, indices)
        len = length(indices);
        data_batch = data(:,indices);
        g.D = cell(K,1);
        lik = zeros(K, len); % a K-by-N matrix with p_k q_k(x_i)
        Sigma_inv = cell(K,1);
        for k = 1:K
            [Sinv, R, Rinv] = invmat(theta.D{k});
            u = sum((Rinv' * data_batch).^2, 1); 
            temp = const * exp(log(theta.p(k)) - sum(log(diag(R))) - 0.5 * u);            
            lik(k, :) = temp;
            Sigma_inv{k} = Sinv;
        end
        component_weights = lik ./ sum(lik, 1);
         
        for k = 1:K
            % Multiply data by the square root of the weight, follows from
            % mvn2factory line 175 - 179
            weight = component_weights(k,:);
            sumW = sum(weight, 2);
            weight = sqrt(weight);            
            data_k = bsxfun(@times, data_batch, weight);
            DDT = (data_k * data_k.');
            Sinv = Sigma_inv{k};
            
            g.D{k} = - 1/(2*len) * ( -sumW * Sinv + Sinv * DDT *Sinv);
        end
        g.p = -1/len * squeeze(sum(component_weights, 2)./theta.p);
    end
    

    
    % ========= problem structure for BW metric ========
    problemBW.M = M_bw;
    problemBW.cost = @costBW;
    problemBW.egrad = @egradBW;
    problemBW.partialegrad = @partialegradBW;
    problemBW.ncostterms = N;
    
    const = (2 * pi)^(-d/2) * exp(1/2);
    
    function f = costBW(theta)        
        f = 0;        
        for k = 1:K
            [~, R, ~] = invmat(theta.D{k});
            % diag(X' C X)
            u = sum((R * data).^2, 1); 
            % first take log then exp is faster than calcualte det
            f = f + exp(log(theta.p(k)) + sum(log(diag(R))) - 0.5 * u);
        end 
        f = - 1/N * sum(log(const * f));
    end

    function g = egradBW(theta)
        g.D = cell(K,1);
        lik = zeros(K, N); % a K-by-N matrix with p_k q_k(x_i)
        Sigma_inv = cell(K,1);
        for k = 1:K
            [Sinv, R, ~] = invmat(theta.D{k});
            u = sum((R * data).^2, 1); 
            temp = const * exp(log(theta.p(k)) + sum(log(diag(R))) - 0.5 * u);            
            lik(k, :) = temp;
            Sigma_inv{k} = Sinv;
        end
        component_weights = lik ./ sum(lik, 1);
         
        for k = 1:K
            % Multiply data by the square root of the weight, follows from
            % mvn2factory line 175 - 179
            weight = component_weights(k,:);
            sumW = sum(weight, 2);
            weight = sqrt(weight);            
            data_k = bsxfun(@times, data, weight);
            DDT = (data_k * data_k.');
            Sinv = Sigma_inv{k};
            
            g.D{k} = - 1/(2*N) * ( sumW * Sinv -  DDT);
        end
                
        g.p = -1/N * squeeze(sum(component_weights, 2)./theta.p);
    end
    
    
    function g = partialegradBW(theta, indices)
        len = length(indices);
        data_batch = data(:,indices);
        g.D = cell(K,1);
        lik = zeros(K, len); % a K-by-N matrix with p_k q_k(x_i)
        Sigma_inv = cell(K,1);
        for k = 1:K
            [Sinv, R, ~] = invmat(theta.D{k});
            u = sum((R * data_batch).^2, 1); 
            temp = const * exp(log(theta.p(k)) + sum(log(diag(R))) - 0.5 * u);            
            lik(k, :) = temp;
            Sigma_inv{k} = Sinv;
        end
        component_weights = lik ./ sum(lik, 1);
         
        for k = 1:K
            % Multiply data by the square root of the weight, follows from
            % mvn2factory line 175 - 179
            weight = component_weights(k,:);
            sumW = sum(weight, 2);
            weight = sqrt(weight);            
            data_k = bsxfun(@times, data_batch, weight);
            DDT = (data_k * data_k.');
            Sinv = Sigma_inv{k};
            
            g.D{k} = - 1/(2*len) * ( sumW * Sinv -  DDT);
        end
        g.p = -1/len * squeeze(sum(component_weights, 2)./theta.p);
    end
    
    
    % ====== problem structure for GBW metric ========
    mbw = sympositivedefiniteBWfactory(d+1);
    mbw.retr = @retr_gbw;
    mbw.inner = @(X, eta, zeta) trace( (X\eta) * (X\zeta) );
    mbw.norm = @(X, eta) real(sqrt(trace( (X\eta) * (X\eta) )));
    powerSPDgbw = powermanifold(mbw, K);
    M_gbw = productmanifold(struct('D',powerSPDgbw ,'p', simplex));
    problemGBW.M = M_gbw;
    problemGBW.cost = @costBW;
    problemGBW.grad = @rgrad_gbw;
    problemGBW.egradBW = @egradBW;
    problemGBW.partialgrad = @partialgradGBW;
    problemGBW.ncostterms = N;
    
    function g = rgrad_gbw(theta)
        g.D = cell(K,1);
        lik = zeros(K, N); % a K-by-N matrix with p_k q_k(x_i)
        %Sigma_inv = cell(K,1);
        for k = 1:K
            [~, R, ~] = invmat(theta.D{k});
            u = sum((R * data).^2, 1); 
            temp = const * exp(log(theta.p(k)) + sum(log(diag(R))) - 0.5 * u);            
            lik(k, :) = temp;
            %Sigma_inv{k} = Sinv;
        end
        component_weights = lik ./ sum(lik, 1);
         
        for k = 1:K
            % Multiply data by the square root of the weight, follows from
            % mvn2factory line 175 - 179
            weight = component_weights(k,:);
            sumW = sum(weight, 2);
            weight = sqrt(weight);            
            data_k = bsxfun(@times, data, weight);
            DDT = (data_k * data_k.');
            %Sinv = Sigma_inv{k};
            
            g.D{k} = - 1/(2*N) * ( sumW * 4*theta.D{k} -  4*theta.D{k}*DDT*theta.D{k});
        end
                
        g.p = -1/N * squeeze(sum(component_weights, 2)./theta.p);
        g.p = simplex.egrad2rgrad(theta.p, g.p);
    end
    

    function g=partialgradGBW(theta, indices)
        len = length(indices);
        data_batch = data(:,indices);
        g.D = cell(K,1);
        lik = zeros(K, len); % a K-by-N matrix with p_k q_k(x_i)
        %Sigma_inv = cell(K,1);
        for k = 1:K
            [Sinv, R, ~] = invmat(theta.D{k});
            u = sum((R * data_batch).^2, 1); 
            temp = const * exp(log(theta.p(k)) + sum(log(diag(R))) - 0.5 * u);            
            lik(k, :) = temp;
            %Sigma_inv{k} = Sinv;
        end
        component_weights = lik ./ sum(lik, 1);
         
        for k = 1:K
            % Multiply data by the square root of the weight, follows from
            % mvn2factory line 175 - 179
            weight = component_weights(k,:);
            sumW = sum(weight, 2);
            weight = sqrt(weight);            
            data_k = bsxfun(@times, data_batch, weight);
            DDT = (data_k * data_k.');
            %Sinv = Sigma_inv{k};
            
            g.D{k} = - 1/(2*len) * ( sumW * 4*theta.D{k} -  4*theta.D{k}*DDT*theta.D{k});
        end
        g.p = -1/len * squeeze(sum(component_weights, 2)./theta.p);
        g.p = simplex.egrad2rgrad(theta.p, g.p);
    end
    
    
    function Y = retr_gbw(X, eta, t)
        if nargin < 3
            t = 1.0;
        end
        [invX, ~,~] = invmat(X);
        teta = t*eta;
        Y = X + teta + 0.25* teta* invX * teta;
    end
    
    
    
    % ======= problem structure for LE metric ========
    % Here we use the parameterization of BW to avoid inversion as much as
    % possible
    problemLE.M = M_le;
    problemLE.cost = @costLE;
    problemLE.egrad = @egradLE;
    problemLE.partialegrad = @partialegradLE;
    problemLE.ncostterms = N;

    const = (2 * pi)^(-d/2) * exp(1/2);
    
    function f = costLE(theta)        
        f = 0;        
        for k = 1:K
            % diag(X' C X)
            u = diag(data' * expm(theta.D{k}) * data);
            % log det exp equals trace
            f = f + exp(log(theta.p(k)) + 0.5*trace(theta.D{k}) - 0.5 * u);
        end 
        f = - 1/N * sum(log(const * f));
    end

    function g = egradLE(theta)
        g.D = cell(K,1);
        lik = zeros(K, N); % a K-by-N matrix with p_k q_k(x_i)
        for k = 1:K
            u = diag(data' * expm(theta.D{k}) * data);
            temp = const * exp(log(theta.p(k)) + 0.5*trace(theta.D{k}) - 0.5 * u);            
            lik(k, :) = temp;
        end
        component_weights = lik ./ sum(lik, 1);
         
        for k = 1:K
            % Multiply data by the square root of the weight, follows from
            % mvn2factory line 175 - 179
            weight = component_weights(k,:);
            sumW = sum(weight, 2);
            weight = sqrt(weight);            
            data_k = bsxfun(@times, data, weight);
            DDT = (data_k * data_k.');
            
            g.D{k} = - 1/(2*N) * ( sumW * eye(d+1) - symm(dexpm(theta.D{k}, DDT)));
        end
                
        g.p = -1/N *squeeze(sum(component_weights, 2)./theta.p);
    end

    function g = partialegradLE(theta, indices)
        len = length(indices);
        data_batch = data(:,indices);
        g.D = cell(K,1);
        lik = zeros(K, len); % a K-by-N matrix with p_k q_k(x_i)
        for k = 1:K
            u = diag(data_batch' * expm(theta.D{k}) * data_batch);
            temp = const * exp(log(theta.p(k)) + 0.5*trace(theta.D{k}) - 0.5 * u);            
            lik(k, :) = temp;
        end
        component_weights = lik ./ sum(lik, 1);
         
        for k = 1:K
            % Multiply data by the square root of the weight, follows from
            % mvn2factory line 175 - 179
            weight = component_weights(k,:);
            sumW = sum(weight, 2);
            weight = sqrt(weight);            
            data_k = bsxfun(@times, data_batch, weight);
            DDT = (data_k * data_k.');
            
            g.D{k} = - 1/(2*len) * (sumW * eye(d+1) - symm(dexpm(theta.D{k}, DDT)) );
        end
        g.p = -1/len * squeeze(sum(component_weights, 2)./theta.p);
    end
    
    
    %%
    % === init (kmeans++) ===
    num_init = 20;
    best_loss = Inf;
    w = ones(1,K)/K;
    for ii = 1:num_init
        [mu, sigmaEM] = kmeans_init(data_org, K, 'sqeuclidean');
        try chol(sigmaEM)
            
        catch ME
            sigmaEM = sigmaEM + 1e-6 * eye(d);
        end
        loss = costEM(mu, sigmaEM, w);
        if loss < best_loss
            best_mu = mu;
            best_sigma = sigmaEM;
            best_loss = loss;
        end
    end
    
    
    % AI
    for j = 1:K
        x0.D{j} = main2new(best_mu(:,j), best_sigma(:,:,j));
    end
    x0.p = w';
        
    % BW
    for j = 1:K
        bw_init.D{j} = invmat(x0.D{j});
    end
    bw_init.p = x0.p;
    
    % GBW
    for j = 1:K
        gbw_init.D{j} = invmat(x0.D{j});
    end
    gbw_init.p = x0.p;
    
    % LE
    for j = 1:K
        le_init.D{j} = logm(bw_init.D{j});
    end
    le_init.p = x0.p;
    
    
    function stats = computestats(problem, theta, stats)
        g = problem.egrad(theta);
        egradnorm = 0;
        for i = 1:K
            egradnorm = egradnorm + norm(theta.D{i}*symm(g.D{i}), 'fro');
        end
        stats.egradnorm = egradnorm/K;
    end

    
    function stats = computestatsGBW(problem, theta, stats)
        g = problem.egradBW(theta);
        egradnorm = 0;
        for i = 1:K
            egradnorm = egradnorm + norm(theta.D{i}*symm(g.D{i}), 'fro');
        end
        stats.egradnorm = egradnorm/K;
    end
    
    
    function stats = computestatsLE(problem, theta, stats)
        theta_temp = theta;
        for i = 1:K
            theta_temp.D{i} = expm(theta.D{i});
        end
        g = egradBW(theta_temp);
        egradnorm = 0;
        for i = 1:K
            egradnorm = egradnorm + norm(theta_temp.D{i}*symm(g.D{i}), 'fro');
        end
        stats.egradnorm = egradnorm/K;
    end
    
            

    %% solve
    
    options.statsfun = @computestats;
    

    if strcmp(solver_choice, 'SGD')
        options.batchsize = 50; 
        options.stepsize_type = 'decay';
        options.maxiter = 3000;
        
        switch data_choice
            case 'iris'
                % Iris
                ai_init_lr = 2;
                bw_init_lr = 0.001;
                gbw_init_lr = 0.5;
                le_init_lr = 5e-4;
            case 'kmeans'
                % kmeans
                ai_init_lr = 1;
                bw_init_lr = 0.005;
                gbw_init_lr = 0.05;
                le_init_lr = 0.01;
                
        end
        options.stepsize_init = ai_init_lr;
        [~, info_AI, ~] = stochasticgradient_mod(problemAI, x0, options);
        
        options.stepsize_init = bw_init_lr;
        [~, info_BW, ~] = stochasticgradient_mod(problemBW, bw_init, options);
        
        options.stepsize_init = gbw_init_lr;
        options.statsfun = @computestatsGBW;
        [~, info_GBW, ~] = stochasticgradient_mod(problemGBW, gbw_init, options);
        
        options.stepsize_init = le_init_lr;
        options.statsfun = @computestatsLE;
        [~, info_LE, ~] = stochasticgradient_mod(problemLE, le_init, options);
    end
    
            
    %% plots 
    lw = 4.0;
    ms = 7.0;
    fs = 25;    
  
        
    % cost
    h1 = figure(1);
    plot(221);
    semilogy([info_BW.iter], [info_BW.cost], '-o',  'LineWidth', lw, 'MarkerSize',ms);  hold on;
    semilogy([info_AI.iter], [info_AI.cost], '-+',  'LineWidth', lw, 'MarkerSize',ms);  hold on;
    semilogy([info_LE.iter], [info_LE.cost], '-x',  'LineWidth', lw, 'MarkerSize',ms);  hold on;
    semilogy([info_GBW.iter], [info_GBW.cost], '-o',  'LineWidth', lw, 'MarkerSize',ms);  hold on;
    hold off;
    ax1 = gca;
    set(ax1,'FontSize', fs);
    set(h1,'Position',[100 100 600 500]);
    xlabel('Iterations', 'fontsize', fs);
    ylabel('Loss', 'fontsize', fs);
    legend(['BW: ',num2str(bw_init_lr)], ['AI: ',num2str(ai_init_lr)], ['LE: ',num2str(le_init_lr)], ['GBW: ', num2str(gbw_init_lr)]);


    % egrad
    h2 = figure(2);
    plot(221);
    semilogy([info_BW.iter], [info_BW.egradnorm], '-o',  'LineWidth', lw, 'MarkerSize',ms);  hold on;
    semilogy([info_AI.iter], [info_AI.egradnorm], '-+', 'LineWidth', lw, 'MarkerSize',ms);  hold on;
    semilogy([info_LE.iter], [info_LE.egradnorm], '-x',  'LineWidth', lw, 'MarkerSize',ms);  hold on;
    semilogy([info_GBW.iter], [info_GBW.egradnorm], '-o', 'LineWidth', lw, 'MarkerSize',ms);  hold on;
    hold off;
    ax1 = gca;
    set(ax1,'FontSize', fs);
    set(h2,'Position',[100 100 600 500]);
    xlabel('Iterations', 'fontsize', fs);
    ylabel('Euclidean grad norm', 'fontsize', fs);
    legend(['BW: ',num2str(bw_init_lr)], ['AI: ',num2str(ai_init_lr)], ['LE: ',num2str(le_init_lr)], ['GBW: ',num2str(gbw_init_lr)]);
    
    % dist-time
    h3 = figure(3);
    plot(221);
    semilogy([info_BW.time], [info_BW.egradnorm], '-o',  'LineWidth', lw, 'MarkerSize',ms);  hold on;
    semilogy([info_AI.time], [info_AI.egradnorm], '-+',  'LineWidth', lw, 'MarkerSize',ms);  hold on;
    semilogy([info_LE.time], [info_LE.egradnorm], '-x',  'LineWidth', lw, 'MarkerSize',ms);  hold on;
    semilogy([info_GBW.time], [info_GBW.egradnorm], '-o', 'LineWidth', lw, 'MarkerSize',ms);  hold on;
    hold off;
    ax1 = gca;
    xlim([0 2]);
    set(ax1,'FontSize', fs);
    set(h3,'Position',[100 100 600 500]);
    xlabel('Time (s)', 'fontsize', fs);
    ylabel('Euclidean grad norm', 'fontsize', fs);
        
    
    %% helper functions
    % invert a spd matrix with chol
    function [invSigma, R, Rinv] = invmat(sigma)
        
        [R, p] = chol(sigma); % C=R' R ( R upper trangular)
        if p > 0
            Rinv = zeros(d+1);
            R = Inf(d+1);
            warning('matrix is not symmetric positive definite ...\n');
        else
            Rinv = R \ eye(d+1); 
        end
        invSigma = Rinv * Rinv';
        
    end
    
    % convert from transformed sigma to mu and sigma for EM
    % NOTE: input should be S not S inverse
    function [mu, sigmaEM] = new2main(sigma)
        
        sigma_tmp = sigma(1:d,1:d);
        sigmaa = sigma(1:d,end);
        sigmab = sigma(end,end);
        sigmaEM = sigma_tmp - sigmaa*sigmaa.' / sigmab;
        mu = sigmaa/ sigmab;
        
    end

    % convert from mu and sigma back to new transformed covariance matrix
    % NOTE: output is S not S inverse
    function sigma = main2new(mu, sigmaEM)
        
        sigma = zeros(d+1);
        sigma(end,1:d) = mu;
        sigma(1:d,end) = mu.';
        sigma(1:d,1:d) = sigmaEM + mu*mu.';
        sigma(end,end) = 1;
            
    end

    function llh = costEM(mu, sigma, w)
        % Calculate llh based on mu and sigma and w
        R = zeros(N,K);
        for i = 1:K
            R(:,i) = loggausspdf(data_org,mu(:,i),sigma(:,:,i));
        end
        R = bsxfun(@plus,R,log(w));
        T = logsumexp(R,2);
        llh = -sum(T)/N; 
        
    end

end

