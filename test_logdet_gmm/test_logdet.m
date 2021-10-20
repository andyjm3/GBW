function [] = test_logdet()

    % Test on log-det optimization 
    % as in http://www.optimization-online.org/DB_FILE/2009/09/2409.pdf
    % Loss: <C, S> - logdet(S)

    % All saved figures for main text use rng(0).
    rng('default');
    rng(0);
        
    symm = @(Z) 0.5*(Z+Z');
    
    %% data
    
    data_choice = 'Ex1';
    CN = 10;
        
    switch data_choice
        % Example 1:
        case 'Ex1'
            d = 100;
            D = 1000*diag(logspace(-log10(CN), 0, d)); fprintf('Exponential decay of singular values with CN %d.\n \n\n', CN);
            % D = diag(1:(CN-1)/(n -1):CN); fprintf('Linear decay of singular values with CN %d.\n \n\n', CN); % Linear decay
            [Q, R] = qr(randn(d)); %#ok
            S = Q*D*Q';
            invS = invmat(S);
    end
    
    C = invS;
    
    %% problem define
    
    problemAI.M = sympositivedefinitefactory(d);
    problemAI.cost = @cost;
    problemAI.egrad = @egrad;
    problemAI.ehess = @ehess;
    
    problemBW.M = sympositivedefiniteBWfactory(d);
    problemBW.cost = @cost;
    problemBW.egrad = @egrad;
    problemBW.ehess = @ehess;
    
    function f = cost(X)
        f = trace(C * X) - log(det(X));
    end
    
    function g = egrad(X)
        [invX, ~,~] = invmat(X);
        g = C - invX;
    end

    function h = ehess(X, Xdot)
        [invX, ~,~] = invmat(X);
        h = invX * Xdot * invX;
    end

    function mydist = disttosol(X)
        mydist = norm(X - S, 'fro');
    end

    function stats = computestats(problem, X, stats)
        stats.disttosol = disttosol(X);
    end


    % only changing the retraction, rgrad and rhess
    problemGBW.M = sympositivedefiniteBWfactory(d);
    problemGBW.cost = @cost;
    problemGBW.grad = @rgrad_gbw;
    problemGBW.hess = @rhess_gbw;
    problemGBW.M.retr = @retr_gbw;
    problemGBW.M.inner = @(X, eta, zeta) trace( (X\eta) * (X\zeta) );
    problemGBW.M.norm = @(X, eta) real(sqrt(trace( (X\eta) * (X\eta) )));
    
    function g = rgrad_gbw(X)
        %[invX, ~,~] = invmat(X);
        g = 4*X*C*X - 4*X;
    end

    function h = rhess_gbw(X, Xdot)
        %[invX, ~,~] = invmat(X);
        %ehess = invX * Xdot * invX;
        
        %h = 4*symm(Xdot) + 2*symm(X * C * Xdot - Xdot) ...
        %    + 2*symm( X * symm( C*Xdot*invX - ehess) * X ) ...
        %    - symm(2*Xdot * (C*X - eye(d)));
        
        h = 2*symm(Xdot) + 2* symm(X *C* Xdot); % the same as above
    end 

    function Y = retr_gbw(X, eta, t)
        if nargin < 3
            t = 1.0;
        end
        [invX, ~,~] = invmat(X);
        teta = t*eta;
        Y = X + teta + 0.25* teta* invX * teta;
    end

    
    % for LE
    problemLE.M = symmetricfactory(d);
    problemLE.cost = @costLE;
    problemLE.egrad = @egradLE;
    
    function f = costLE(S)
        f = trace(C * expm(S)) - trace(S);
    end

    function g = egradLE(S)
        g = symm(dexpm(S, C)) - eye(d);
    end

    function stats = computestatsLE(problem, S, stats)
        stats.disttosol = disttosol(expm(S));
    end   
    
    
    %% solve
    options.maxiter = 7;
    options.tolgradnorm = 1e-10;
    
    Xinitial = eye(d);
    Sinitial = logm(Xinitial);
    
    options.statsfun = @computestats;        
    [~, ~, info_AI, ~] = trustregions(problemAI, Xinitial, options);
    [~, ~, info_BW, ~] = trustregions(problemBW, Xinitial, options);  
    options.maxiter = 10;
    [~, ~, info_GBW, ~] = trustregions(problemGBW, Xinitial, options);  

    options.statsfun = @computestatsLE;
    options.maxiter = 8;
    [~, ~, info_LE, ~] = trustregions(problemLE, Sinitial, options);
    
    
    %% plots 
    lw = 4.0;
    ms = 7.0;
    fs = 25;
    colors = {[55, 126, 184]/255, [228, 26, 28]/255, [247, 129, 191]/255, ...
              [166, 86, 40]/255, [255, 255, 51]/255, [255, 127, 0]/255, ...
              [152, 78, 163]/255, [77, 175, 74]/255}; 
    
          
    
    
    % cost
    h1 = figure(1);
    plot(221);
    plot(cumsum([info_BW.numinner]), [info_BW.cost], '-o',  'LineWidth', lw, 'MarkerSize',ms);  hold on;
    plot(cumsum([info_AI.numinner]), [info_AI.cost], '-+',  'LineWidth', lw, 'MarkerSize',ms);  hold on;
    plot(cumsum([info_LE.numinner]), [info_LE.cost], '-x',  'LineWidth', lw, 'MarkerSize',ms);  hold on;
    plot(cumsum([info_GBW.numinner]), [info_GBW.cost], '-o', 'LineWidth', lw, 'MarkerSize',ms);  hold on;
    hold off;
    ax1 = gca;
    set(ax1,'FontSize', fs);
    set(h1,'Position',[100 100 600 500]);
    xlabel('Inner iterations (cumsum)', 'fontsize', fs);
    ylabel('Loss', 'fontsize', fs);
        
        
    % dist
    h2 = figure(2);
    plot(221);
    semilogy(cumsum([info_BW.numinner]), [info_BW.disttosol], '-o', 'LineWidth', lw, 'MarkerSize',ms);  hold on;
    semilogy(cumsum([info_AI.numinner]), [info_AI.disttosol], '-+', 'LineWidth', lw, 'MarkerSize',ms);  hold on;
    semilogy(cumsum([info_LE.numinner]), [info_LE.disttosol], '-x', 'LineWidth', lw, 'MarkerSize',ms);  hold on;
    semilogy(cumsum([info_GBW.numinner]), [info_GBW.disttosol], '-o', 'LineWidth', lw, 'MarkerSize',ms);  hold on;
    hold off;
    ax1 = gca;
    set(ax1,'FontSize', fs);
    set(h2,'Position',[100 100 600 500]);
    xlabel('Inner iterations (cumsum)', 'fontsize', fs);
    ylabel('Distance to solution', 'fontsize', fs);
    legend('BW', 'AI', 'LE', 'GBW')
       
    % dist-time
    h3 = figure(3);
    plot(221);
    semilogy([info_BW.time], [info_BW.disttosol], '-o',  'LineWidth', lw, 'MarkerSize',ms);  hold on;
    semilogy([info_AI.time], [info_AI.disttosol], '-+',  'LineWidth', lw, 'MarkerSize',ms);  hold on;
    semilogy([info_LE.time], [info_LE.disttosol], '-x',  'LineWidth', lw, 'MarkerSize',ms);  hold on;
    semilogy([info_GBW.time], [info_GBW.disttosol], '-o', 'LineWidth', lw, 'MarkerSize',ms);  hold on;
    hold off;
    ax1 = gca;
    set(ax1,'FontSize', fs);
    set(h3,'Position',[100 100 600 500]);
    xlabel('Time (s)', 'fontsize', fs);
    ylabel('Distance to solution', 'fontsize', fs);
    
    %% helper function
    function [invSigma, R, Rinv] = invmat(sigma)
        
        [R, p] = chol(sigma); % C=R' R ( R upper trangular)
        if p > 0
            Rinv = zeros(d);
            R = Inf(d);
            warning('matrix is not symmetric positive definite ...\n');
        else
            Rinv = R \ eye(d); 
        end
        invSigma = Rinv * Rinv';
        
    end

end

