% data(cellfun(@(x) isempty(x), data)) = {NaN};
clear p1 p2 stats_output combined_p
pairs = nchoosek(groups,2);
data_pairs = nchoosek(1:size(data,1),2);

count = 1;
for ii = 1:size(data_pairs,1)
    
    if strcmp(pairs{ii,1}(1:(end-6)), pairs{ii,2}(1:(end-6)))
        data_x = data(data_pairs(ii,1),:);
        data_y = data(data_pairs(ii,2),:);
    
        data_x = data_x(~isnan(data_x));
        data_y = data_y(~isnan(data_y));
    
        % data_x_mean = mean(data_x);
        % data_y_mean = mean(data_y);
    
        [h1(count), p1(count)] = ttest2(data_x, data_y);
    
        [h2(count), p2(count)] = ttest(data_x, data_y);
        
        pairs_output(count,1) = pairs(ii,1);
        pairs_output(count,2) = pairs(ii,2);
        stats_output{count, 1} = p1(count);
        stats_output{count, 2} = p2(count);
        
        count = count + 1;
    end
end

p1 = p1';
p2 = p2';

combined_p = [p1; p2];