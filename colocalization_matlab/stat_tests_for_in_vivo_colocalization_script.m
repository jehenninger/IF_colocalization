pairs = nchoosek(groups,2);
data_pairs = nchoosek(1:size(data,1),2);

for ii = 1:size(data_pairs,1)
    
    [h1(ii), p1(ii)] = ttest2(data(data_pairs(ii,1),:), data(data_pairs(ii,2),:));
    
    [h2(ii), p2(ii)] = ttest(data(data_pairs(ii,1),:), data(data_pairs(ii,2),:));
end

p1 = p1';
p2 = p2';

combined_p = [p1, p2];