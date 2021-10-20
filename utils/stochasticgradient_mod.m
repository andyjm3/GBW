function [x, info, options] = stochasticgradient_mod(problem, x, options)
% Stochastic gradient (SG) minimization algorithm for Manopt.
%
% function [x, info, options] = stochasticgradient(problem)
% function [x, info, options] = stochasticgradient(problem, x0)
% function [x, info, options] = stochasticgradient(problem, x0, options)
% function [x, info, options] = stochasticgradient(problem, [], options)
%
% Apply the Riemannian stochastic gradient algorithm to the problem defined
% in the problem structure, starting at x0 if it is provided (otherwise, at
% a random point on the manifold). To specify options whilst not specifying
% an initial guess, give x0 as [] (the empty matrix).
%
% The problem structure must contain the following fields:
%
%  problem.M:
%       Defines the manifold to optimize over, given by a factory.
%
%  problem.partialgrad or problem.partialegrad (or equivalent)
%       Describes the partial gradients of the cost function. If the cost
%       function is of the form f(x) = sum_{k=1}^N f_k(x),
%       then partialegrad(x, K) = sum_{k \in K} grad f_k(x).
%       As usual, partialgrad must define the Riemannian gradient, whereas
%       partialegrad defines a Euclidean (classical) gradient which will be
%       converted automatically to a Riemannian gradient. Use the tool
%       checkgradient(problem) to check it. K is a /row/ vector, which
%       makes it natural to write for k = K, ..., end.
%
%  problem.ncostterms
%       An integer specifying how many terms are in the cost function (in
%       the example above, that would be N.)
%
% Importantly, the cost function itself needs not be specified.
%
% Some of the options of the solver are specific to this file. Please have
% a look inside the code.
%
% To record the value of the cost function or the norm of the gradient for
% example (which are statistics the algorithm does not require and hence
% does not compute by default), one can set the following options:
%
%   metrics.cost = @(problem, x) getCost(problem, x);
%   metrics.gradnorm = @(problem, x) problem.M.norm(x, getGradient(problem, x));
%   options.statsfun = statsfunhelper(metrics);
%
% Important caveat: stochastic algorithms usually return an average of the
% last few iterates. Computing averages on manifolds can be expensive.
% Currently, this solver does not compute averages and simply returns the
% last iterate. Using options.statsfun, it is possible for the user to
% compute averages manually. If you have ideas on how to do this
% generically, we welcome feedback. In particular, approximate means could
% be computed with M.pairmean which is available in many geometries.
%
% See also: steepestdescent

% This file is part of Manopt: www.manopt.org.
% Original authors: Bamdev Mishra <bamdevm@gmail.com>,
%                   Hiroyuki Kasai <kasai@is.uec.ac.jp>, and
%                   Hiroyuki Sato <hsato@ms.kagu.tus.ac.jp>, 22 April 2016.
% Contributors: Nicolas Boumal
% Change log: 
%
%   06 July 2019 (BM):  
%      Added preconditioner support. This allows to use adaptive algorithms.
%   31 July 2020 (AH):
%      Modify to trace cost and gradnorm for print only.
    
    % for tracing cost and gradnorm only
    if ~canGetCost(problem)
        warning('manopt:getCost', ...
            'No cost provided. The algorithm will likely abort.');
    end
    if ~canGetGradient(problem) && ~canGetApproxGradient(problem)
        % Note: we do not give a warning if an approximate Hessian is
        % explicitly given in the problem description, as in that case the user
        % seems to be aware of the issue.
        warning('manopt:getGradient:approx', ...
            ['No gradient provided. Using an FD approximation instead (slow).\n' ...
            'It may be necessary to increase options.tolgradnorm.\n' ...
            'To disable this warning: warning(''off'', ''manopt:getGradient:approx'')']);
        problem.approxgrad = approxgradientFD(problem);
    end

    % Verify that the problem description is sufficient for the solver.
    if ~canGetPartialGradient(problem)
        warning('manopt:getPartialGradient', ...
         'No partial gradient provided. The algorithm will likely abort.');
    end
    
   
    % Set local default
    localdefaults.maxiter = 1000;       % Maximum number of iterations
    localdefaults.batchsize = 1;        % Batchsize (# cost terms per iter)
    localdefaults.verbosity = 2;        % Output verbosity (0, 1 or 2)
    localdefaults.storedepth = 20;      % Limit amount of caching
    localdefaults.transport = 'ret_vector';         % Defualt transport type
    
    % Check stopping criteria and save stats every checkperiod iterations.
    localdefaults.checkperiod = 100;
    
    % stepsizefun is a function implementing a step size selection
    % algorithm. See that function for help with options, which can be
    % specified in the options structure passed to the solver directly.
    localdefaults.stepsizefun = @stepsize_sg;
    
    % Merge global and local defaults, then merge w/ user options, if any.
    localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(localdefaults, options);
    
    
    assert(options.checkperiod >= 1, ...
                 'options.checkperiod must be a positive integer (>= 1).');
    
    
    % If no initial point x is given by the user, generate one at random.
    if ~exist('x', 'var') || isempty(x)
        x = problem.M.rand();
    end
    
    % Create a store database and get a key for the current x
    storedb = StoreDB(options.storedepth);
    key = storedb.getNewKey();
    
    
    % Elapsed time for the current set of iterations, where a set of
    % iterations comprises options.checkperiod iterations. We do not
    % count time spent for such things as logging statistics, as these are
    % not relevant to the actual optimization process.
    elapsed_time = 0;
    
    % Total number of completed steps
    iter = 0;
    % Total number of sfo
    gradcnt = 0;
    
    % Total number of saved stats at this point.
    savedstats = 0;
    
    % Collect and save stats in a struct array info, and preallocate.
    stats = savestats();
    info(1) = stats;
    savedstats = savedstats + 1;
    if isinf(options.maxiter)
        % We trust that if the user set maxiter = inf, then they defined
        % another stopping criterion.
        preallocate = 1e5;
    else
        preallocate = ceil(options.maxiter / options.checkperiod) + 1;
    end
    info(preallocate).iter = [];
    
    mode = 'RSGD';
    if options.verbosity >= 2
       fprintf('\n-------------------------------------------------------\n');
       fprintf('%s:  iter\t               cost val\t   stepsize\n', mode);
       fprintf('%s:  %5d\t%+.16e\t%.8e\n', mode, 0, stats.cost, NaN);        
    end
    
    
    % Main loop.
    stop = false;
    while iter < options.maxiter
        
        % Record start time.
        start_time = tic();
        
        % Draw the samples with replacement.
        idx_batch = randi(problem.ncostterms, options.batchsize, 1);
        
        % Compute partial gradient on this batch.
        pgrad = getPartialGradient(problem, x, idx_batch, storedb, key);
        gradcnt = gradcnt + options.batchsize;

        % Apply preconditioner to the partial gradient.
        Ppgrad = getPrecon(problem, x, pgrad, storedb, key);
        
        % Compute a step size and the corresponding new point x.
        [stepsize, newx, newkey, ssstats] = ...
                           options.stepsizefun(problem, x, Ppgrad, iter, ...
                                               options, storedb, key);
        
        % Make the step: transfer iterate, remove cache from previous x.
        storedb.removefirstifdifferent(key, newkey);
        x = newx;
        key = newkey;
        
        % Make sure we do not use too much memory for the store database.
        storedb.purge();
        
        % Total number of completed steps.
        iter = iter + 1;
        
        % Elapsed time doing actual optimization work so far in this
        % set of options.checkperiod iterations.
        elapsed_time = elapsed_time + toc(start_time);
        
        
        % Check stopping criteria and save stats every checkperiod iters.
        if mod(iter, options.checkperiod) == 0
            
            % Log statistics for freshly executed iteration.
            stats = savestats();
            info(savedstats+1) = stats;
            savedstats = savedstats + 1;
            
            % Reset timer.
            elapsed_time = 0;
            
            % Print output.
            if options.verbosity >= 2
                fprintf('%s:  %5d\t%+.16e\t%.8e\n', mode, iter, stats.cost, stepsize);                    
            end
            
            % Run standard stopping criterion checks.
            [stop, reason] = stoppingcriterion(problem, x, ...
                                               options, info, savedstats);
            if stop
                if options.verbosity >= 1
                    fprintf([reason '\n']);
                end
                break;
            end
        
        end

    end
    
    
    % Keep only the relevant portion of the info struct-array.
    info = info(1:savedstats);
    
    
    % Display a final information message.
    if options.verbosity >= 1
        if ~stop
            % We stopped not because of stoppingcriterion but because the
            % loop came to an end, which means maxiter triggered.
            msg = 'Max iteration count reached; options.maxiter = %g.\n';
            fprintf(msg, options.maxiter);
        end
        fprintf('Total time is %f [s] (excludes statsfun)\n', ...
                info(end).time + elapsed_time);
    end
    
    
    % Helper function to collect statistics to be saved at
    % index checkperiodcount+1 in info.
    function stats = savestats()
        stats.iter = iter;
        
        % Compute Riemannian cost and gradient on this batch.
        cost = getCost(problem, x, storedb, key);
        
        if savedstats == 0
            stats.time = 0;
            stats.stepsize = NaN;
            stats.stepsize_stats = [];
            stats.cost = cost;
            stats.gradcnt = gradcnt;
        else
            stats.time = info(savedstats).time + elapsed_time;
            stats.stepsize = stepsize;
            stats.stepsize_stats = ssstats;
            stats.cost = cost;
            stats.gradcnt = gradcnt;
        end
        stats = applyStatsfun(problem, x, storedb, key, options, stats);
    end
    
end
