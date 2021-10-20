function [ data_new, index ] = shuffle_data( data )
[~,~,N] = size(data);

index = randperm(N);
data_new = data(:,:,index);
end

