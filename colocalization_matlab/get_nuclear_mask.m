function [mask_output] = get_nuclear_mask(mask_file)
%mask_file is an absolute path to a multiimage TIFF mask channel
%(usually DAPI)

% mask_output is a cell array of binary masks, where each row
% corresponds to a z-slice
top_z_slices = 10;
mask_info = imfinfo(mask_file);
[mask_num_of_images, ~] = size(mask_info);

focus_test_value = zeros(mask_num_of_images, 1);

dummy_mask = false(mask_info(1).Width, mask_info(1).Height); % will use a completely black/0 mask for z-slices that are out of focus

% find z slices in focus
for jj = 1:mask_num_of_images
    
    mask_image = imread(mask_file, jj);
    focus_test = imgradient(mask_image);
    focus_test_value(jj) = mean(focus_test(:)); % this is a good metric for the 'focus' of an image
    
%             disp(['Zstack ', num2str(jj), 'Focus test value ', num2str(focus_test(jj))]);
%     
%             plot(jj, focus_test(jj),'o-');
%             hold on
end

[~, images_in_focus] = maxk(focus_test_value, top_z_slices); % arbitrarily picking the top 10 z-slices in terms of focus


% generate binary masks only for z slices in focus
mask_output = cell(mask_num_of_images, 1);
for ii = 1:mask_num_of_images
   
    if ismember(ii, images_in_focus)
        mask_image = imread(mask_file, ii);
        mask_bw{ii} = imbinarize(mask_image);
        mask_output{ii} = imfill(mask_bw{ii}, 'holes'); %fill in holes of nuclei
        
        %figure, montage({double(mask_bw{ii}), double(mask_output{ii})});
    else
        mask_output{ii} = dummy_mask;
    end
    
end

% figure, montage({double(mask_bw), double(mask_output)});

end

