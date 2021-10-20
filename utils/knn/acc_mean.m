function [mean_acc, std_acc] = acc_mean(final_result, r_list)
% compute average acc based on the final result

N_repeat = length(final_result);
r_len = length(r_list);

final_result_table = zeros(N_repeat, r_len, 1); % only for BW, LE, AI
for ii = 1:N_repeat
    bw = final_result{ii}.bw;
    %ai = final_result{ii}.ai;
    %le = final_result{ii}.le;
    
    assert(length(bw) == r_len);
    %assert(length(ai) == r_len);
    %assert(length(le) == r_len);
    
    for jj = 1:r_len
        final_result_table(ii, jj, 1) = bw(jj).Accuracy;
        %final_result_table(ii, jj, 2) = le(jj).Accuracy;
        %final_result_table(ii, jj, 3) = ai(jj).Accuracy;
    end
    
end

mean_acc = mean(final_result_table, 1);
std_acc = std(final_result_table, 0, 1);

end

