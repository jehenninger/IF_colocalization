function [rho] = testForColoc(channel_1_file, channel_2_file, mask_array, draw_graph, graph_title)

% mask_array is the output of get_nuclear_mask. It is a cell array of
% nuclei masks of all z-slices, but only 10 of the most in-focus z-slices
% actually have any 1's.

if ~exist('draw_graph','var')
    draw_graph = 0;
end

channel_1_info = imfinfo(channel_1_file);
[channel_1_num_of_images, ~] = size(channel_1_info);

[~, channel_1_name, ~] = fileparts(channel_1_file);
[~, channel_2_name, ~] = fileparts(channel_2_file);


for jj = 1:channel_1_num_of_images
    
    mask_image = mask_array{jj};
    
    temp_image_1 = imread(channel_1_file, jj);
    temp_image_2 = imread(channel_2_file, jj);
    
    
    % Show masked image.
    % masked_channel_image = bsxfun(@times, temp_image_1, cast(mask_image,class(temp_image_1)));
    % figure, imshow(masked_channel_image,[]);
    
    channel_1_mask_pixels = temp_image_1(mask_image);
    channel_2_mask_pixels = temp_image_2(mask_image);
    
    if jj == 1
        total_channel_1_mask_pixels = channel_1_mask_pixels;
        total_channel_2_mask_pixels = channel_2_mask_pixels;
    else
        total_channel_1_mask_pixels = [total_channel_1_mask_pixels; channel_1_mask_pixels];
        total_channel_2_mask_pixels = [total_channel_2_mask_pixels; channel_2_mask_pixels];
    end
    
    
end

total_channel_1_mask_pixels = double(total_channel_1_mask_pixels);
total_channel_2_mask_pixels = double(total_channel_2_mask_pixels);


% channel_1_sample = datasample(total_channel_1_mask_pixels,50000, 'Replace', false);
% channel_2_sample = datasample(total_channel_2_mask_pixels,50000, 'Replace', false);
% 
% norm_channel_1_sample = normalize_channel(channel_1_sample);
% norm_channel_2_sample = normalize_channel(channel_2_sample);

rho = corr([total_channel_1_mask_pixels, total_channel_2_mask_pixels], 'Tail', 'right');
rho = rho(1,2);
disp(['Correlation coefficient is ', num2str(rho)]);

if draw_graph == 1
    figure('Visible','off'),
    b = total_channel_1_mask_pixels\total_channel_2_mask_pixels;
    regLine = b*total_channel_1_mask_pixels;
    plot(total_channel_1_mask_pixels, total_channel_2_mask_pixels,'.');
    hold on
    plot(total_channel_1_mask_pixels, regLine,'r-');
    hold off
    channel_1_name = strrep(channel_1_name, '_', ' ');
    xlabel(channel_1_name);
        
    channel_2_name = strrep(channel_2_name, '_', ' ');
    ylabel(channel_2_name);
    
    graph_title = strrep(graph_title, '_', ' ');
    title(graph_title);
    
    savePath = '/Users/jon/data_analysis/young/20181003_Image_set_for_IF_coloc_from_John/20180914_image_outputs';
    saveas(gcf, fullfile(savePath,[graph_title,'_', channel_1_name, '_v_', channel_2_name, '.png']));
    saveas(gcf, fullfile(savePath,[graph_title,'_', channel_1_name, '_v_', channel_2_name, '.eps']))
    close(gcf);
end


    function normal_channel = normalize_channel(channel)
        channel = double(channel);
        normal_channel = (channel - min(channel(:)))/(max(channel(:))-min(channel(:)));
    end

end
