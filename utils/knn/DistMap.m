function Dmap = DistMap(set1,set2,M,dist)
%DISTMAP calculate the distance map between points in set1 and set2

N1 = size(set1,3);
N2 = size(set2,3);

Dmap = zeros(N1,N2);

switch(dist)
    case 'BW'
        for i=1:N1
            for j=1:N2
                Dmap(i,j) = BWdist(set1(:,:,i), set2(:,:,j),M);
            end
        end
    case 'LE'
        for i=1:N1
            for j=1:N2
                Dmap(i,j) = LEdist(set1(:,:,i), set2(:,:,j),M);
            end
        end  
    case 'AI'
        for i=1:N1
            for j=1:N2
                Dmap(i,j) = AIdist(set1(:,:,i), set2(:,:,j),M);
            end
        end
end
        
end
    

