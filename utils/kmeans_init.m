function [C, sigmas] = kmeans_init(X,k,distance)
    index = zeros(1,k);
    [C(:,1), index(1)] = datasample(X,1,2);
    [d,n] = size(X);
    minDist = inf(n,1);

    % Select the rest of the seeds by a probabilistic model
   for ii = 2:k                    
        minDist = min(minDist,distfun(X,C(:,ii-1),distance));
        denominator = sum(minDist);
        if denominator==0 || isinf(denominator) || isnan(denominator)
            C(:,ii:k) = datasample(X,k-ii+1,2,'Replace',false);
            break;
        end
        sampleProbability = minDist/denominator;
        [C(:,ii), index(ii)] = datasample(X,1,2,'Replace',false,...
            'Weights',sampleProbability);        
   end
   
   % assignment of clusters
   D = distfun(X, C, distance);
   [~, idx] = min(D, [], 2);
   
   % calculate covariance matrix for each cluster
   sigmas = zeros(d,d,k);
   for j = 1:k
       Xj = X(:,idx==j);
       sigmas(:,:,j) = cov(Xj');
   end
   
   
   
   function D = distfun(X, C, dist, iter,rep, reps)
    %DISTFUN Calculate point to cluster centroid distances.

        switch dist
            case 'sqeuclidean'
                D = internal.stats.pdist2mex(X,C,'sqe',[],[],[],[]);  
            case {'cosine','correlation'}
                % The points are normalized, centroids are not, so normalize them
                normC = sqrt(sum(C.^2, 1));
                if any(normC < eps(class(normC))) % small relative to unit-length data points
                    if reps==1
                        error(message('stats:kmeans:ZeroCentroid', iter));
                    else
                        error(message('stats:kmeans:ZeroCentroidRep', iter, rep));
                    end

                end
                C = bsxfun(@rdivide,C,normC);
                D = internal.stats.pdist2mex(X,C,'cos',[],[],[],[]);  
        end
    end % function
   
end

