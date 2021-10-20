function [Y, info] = BWPCA(X, r, args)

X_mean = BWmean(X);

d = size(X, 1);
N = size(X, 3);

problem.M = grassmannfactory(d,r);


% use second-order optimizers
problem.costgrad = @mycostgrad;
function [f, g] = mycostgrad(Y) 
    f = 0; g=0;
    for ii = 1:N
        Xii = X(:,:,ii);
        XiiY = Xii*Y; YtXiiY = Y'*XiiY;
        XbarY = X_mean*Y; YtXbarY = Y'*XbarY;
        
        f = f - trace(YtXiiY + YtXbarY) + 2*trace(real(sqrtm(YtXiiY*YtXbarY)));
        
        YtXiiYhalf = real(sqrtm(YtXiiY));
        
        R = real(dsqrtm(YtXiiYhalf*YtXbarY*YtXiiYhalf, eye(r)));
        g = g - 2* (   XiiY + XbarY - 2*XbarY*(YtXiiYhalf*R*YtXiiYhalf)  - 2*XiiY* dsqrtm(YtXiiY, R*YtXiiYhalf *YtXbarY ) - 2*XiiY* dsqrtm(YtXiiY, YtXbarY*YtXiiYhalf*R )  );
    end
    f = f/(2*N);
    g = g/(2*N);
    g = problem.M.egrad2rgrad(Y, real(g));
end


Y0 = args.init;
options.maxiter = args.maxepoch;
options.tolgradnorm = args.tolgradnorm;

[Y, info] = trustregions(problem, Y0, options);

end

