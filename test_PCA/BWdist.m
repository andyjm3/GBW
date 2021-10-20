function d = BWdist(X,Y, M)
%Compute BW distance with parameter M

symm = @(X) .5*(X+X');

if nargin < 3
    n = size(X,1);
    M = eye(n);
end

Xhalf = sqrtm(X);
d = real(sqrt(trace(M * X) + trace(M * Y) - 2*trace(symm(sqrtm(Xhalf*M*Y*M*Xhalf)))));

end

