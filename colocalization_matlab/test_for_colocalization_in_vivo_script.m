%% NOTES
% Right now, I choose 10 z-slices that are in focus for nuclear masking and
% comparison. Just as a note, the nuclear masking has trouble with nuclei
% not completely in focus or near the edges of the image. Will this affect
% co-localization? Maybe, especially since dapi stains chromatin.

% Bug: There's an issue with 'count()' when finding unique experimental
% group names. It doesn't look for exact match; only partial matches. So
% any groups that contain other groups will give issues. Best for now to
% just give unique names to things.


%% read in metadata file
close all
[metadata_file, metadata_path] = uigetfile('/Users/jon/data_analysis/young/*.xlsx');
metadata_file = fullfile(metadata_path, metadata_file);
%metadata_file = '/Users/jon/data_analysis/young/20181003_Image_set_for_IF_coloc_from_John/2018_06_28/matlab_metadata.xlsx';
[~, ~, metadata] = xlsread(metadata_file);

metadata(cellfun(@(x) isnumeric(x) && isnan(x), metadata)) = {[]};
metadata = metadata(~all(cellfun('isempty', metadata), 2), :); % gets rid of weird Excel artifacts when there are NaN cells

[data_path, ~, ~] = fileparts(metadata_file);

% get unique experimental groups
experimental_groups = metadata(2:end,3);

%% get rho values from replicates and do one-tailed, one-sample t-test
unique_groups = unique(experimental_groups);
num_of_experimental_groups = size(unique_groups, 1);

pval = cell(num_of_experimental_groups, 1);
rho = cell(50, 4);

multicompare = false(num_of_experimental_groups, 1);

rho_count = 0;
pval_count = 1;
for ii = 1:num_of_experimental_groups
    
    rho_count = rho_count + 1;
    
    experimental_group_name = unique_groups{ii};
    
    group_row_idx = find(count(experimental_groups, unique_groups{ii}) == 1);
    group_row_idx = group_row_idx + 1; % because we took away the title row above
    
    group_metadata = metadata(group_row_idx,:);
    
    replicates = cell2mat(group_metadata(:,4));
    
    num_of_replicates = size(unique(replicates),1);
    
    
    
    for jj = 1:num_of_replicates
        replicate_metadata = group_metadata(replicates == jj, :);
        channels = replicate_metadata(:,2);
        
        % find number of channels to compare (for now, we will only support
        % 2 or 3
        
        for kk = 1:size(channels,1)
            channel_test = channels{kk};
            if ~isempty(channel_test) %handles the case where we have removed a NaN
                switch channel_test
                    case 'mask'
                        mask_file = fullfile(data_path, replicate_metadata{kk, 1});
                    case 1
                        channel_1_file = fullfile(data_path, replicate_metadata{kk, 1});
                    case 2
                        channel_2_file = fullfile(data_path, replicate_metadata{kk, 1});
                    case 3
                        channel_3_file = fullfile(data_path, replicate_metadata{kk, 1});
                        multicompare(ii) = true;
                end
            end
        end
        
        mask_array = get_nuclear_mask(mask_file);
        
        if multicompare(ii)
%             if jj == 1
%                 draw_flag = 0; % only draw one replicate
%             else
%                 draw_flag = 0;
%             end
            draw_flag = 1;
%             
            if jj == 1
                rho{rho_count, 1} = [experimental_group_name, '_C1vC2']; %labeling first column
                rho{rho_count+1, 1} = [experimental_group_name, '_C1vC3'];
                rho{rho_count+2, 1} = [experimental_group_name, '_C2vC3'];
            end
            
            rho{rho_count, jj+1} = testForColoc(channel_1_file, channel_2_file, mask_array, draw_flag, experimental_group_name);
            rho_count = rho_count + 1; %these count the rows in the rho output to just concatenate everything
            
            rho{rho_count, jj+1} = testForColoc(channel_1_file, channel_3_file, mask_array, draw_flag, experimental_group_name);
            rho_count = rho_count + 1;
            
            rho{rho_count, jj+1} = testForColoc(channel_2_file, channel_3_file, mask_array, draw_flag, experimental_group_name);
            
            if jj < num_of_replicates % this will fill in the replicates for the multicomparison the correct way
                rho_count = rho_count - 2;
            end
            
            [~, channel_1_name, ~] = fileparts(channel_1_file);
            [~, channel_2_name, ~] = fileparts(channel_2_file);
            [~, channel_3_name, ~] = fileparts(channel_2_file);
            
        else
%             if jj == 1
%                 draw_flag = 0; %only draw one replicate
%             else
%                 draw_flag = 0;
%             end
            
            draw_flag = 1;
            
            rho{rho_count, 1} = [experimental_group_name, '_C1vC2'];
            
            rho{rho_count, jj+1} = testForColoc(channel_1_file, channel_2_file, mask_array, draw_flag, experimental_group_name); %1 to draw graphs
            
            [~, channel_1_name, ~] = fileparts(channel_1_file);
            [~, channel_2_name, ~] = fileparts(channel_2_file);
            
        end
    end
    
    % pval calculation from rho's
    
    % deprecated
    % extract_index_from_cell = @(C,k) cellfun(@(c)c(k), C); %this function will grab a particular index from a matrix in all cells of cell X.
    
    if multicompare(ii)
        pval{pval_count, 1} = [experimental_group_name, '_C1vC2'];
        pval{pval_count+1, 1} = [experimental_group_name, '_C1vC3'];
        pval{pval_count+2, 1} = [experimental_group_name, '_C2vC3'];
        
        % C1vC2
        data_to_test = rho(pval_count,2:end); % this grabs the rho value from every replicate cell
        data_to_test = [data_to_test{:}];
        [~, pval{pval_count, 2}] = ttest(data_to_test);
        pval_count = pval_count + 1;
        
        % C1vC3
        data_to_test = rho(pval_count,2:end);
        data_to_test = [data_to_test{:}];
        [~, pval{pval_count, 2}] = ttest(data_to_test);
        pval_count = pval_count + 1;
        
        % C2vC3
        data_to_test = rho(pval_count,2:end);
        data_to_test = [data_to_test{:}];
        [~, pval{pval_count, 2}] = ttest(data_to_test);
        pval_count = pval_count + 1;
        
    else
        pval{pval_count, 1} = [experimental_group_name, '_C1vC2'];
        data_to_test = rho(pval_count,2:end);
        data_to_test = [data_to_test{:}];
        [~, pval{pval_count,2}] = ttest(data_to_test); % testing that the correlation is not 0. One-sample T-test
        pval_count = pval_count + 1;
    end
    
    
    
end

rho = rho(~all(cellfun('isempty', rho), 2), :); % gets rid of weird Excel artifacts when there are NaN cells
pval = pval(~all(cellfun('isempty', rho), 2), :);

output.Rho = rho;
output.Pval = pval;

