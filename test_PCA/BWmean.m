function A = BWmean(X)
% Compute BW barycenter applying the fixed point iteration
    d = size(X, 1);
    N = size(X, 3);
    maxiter = 80;
    
    A0 = eye(d);
    
    symm = @(X) .5*(X+X');
    
    fprintf('Start computing BW barycenter...\n');
    
    Anew = A0;
    for ii = 1:maxiter
        A = Anew;
        [Asqrt, Asqrtinv] = minvsqrt(A);
                
        mid_term = 0;
        for nn = 1:N
            mid_term = mid_term + real(sqrtm(symm(Asqrt * X(:,:,nn) * Asqrt)));
        end        
        
        Anew = Asqrtinv * mid_term * mid_term * Asqrtinv/(N^2);
        
        fprintf('Norm diff: %4d \n', norm(Anew - A, 'fro'));
        
        if norm(Anew - A, 'fro') < 1e-12
            A = Anew;
            break;
        end
    end
    %A = Anew;
    fprintf('Finish!\n');
        
    % invmat and compute matrix sqrt and inverse
    function [Asqrt, Asqrtinv] = minvsqrt(A)
        [V, D] = eig(A);
        Asqrt = V * diag(real(sqrt(diag(D)))) * V';
        Asqrtinv = V * diag(1./real(sqrt(diag(D)))) * V';
    end   
end

