import os
import re
import numpy as np
import math
from skimage import io, morphology, segmentation, measure

def generate_output(input_params):

    # function that reads in FISH and IF files to generate output centered on
    # various FISH foci and saves all the capture foci, as well as the IF information in all
    # stacks for any given FISH foci

    # get all sub-directories where images are stored
    dir_name = input_params['dir_name']
    p = os.listdir(dir_name)
    p = [s for s in p if re.search('E', s) and os.path.isdir(s)]  # must be directories that have an 'E' in the name

    x_pixel = input_params['x_pixel']
    z_pixel = input_params['z_pixel']

    # store IF and FISH information for each replicate
    sub_dir = list()
    IF_channel = list()
    uniq_folder_id = list()
    FISH_channel = list()

    for count, file in enumerate(p):
        sub_dir[count] = os.path.join(dir_name, file)

        if os.path.isdir(sub_dir[count]):
            local_files = os.listdir(sub_dir[count])

            for j in local_files:
                if re.search(j, input_params['IF_channel']):
                    IF_channel[count] = sub_dir[count] + j
                    uniq_folder_id[count] = file[-2:]

                elif re.search(j, input_params['FISH_channel']):
                    FISH_channel[count] = sub_dir[count] + j

    # get previous called foci if automated foci was note called
    curated_foci_name = list()
    if not input_params['auto_call_foci_flag']:
        p = os.listdir(input_params['csv_folder'])
        for count in range(len(uniq_folder_id)):
            i = 3
            flag = 0
            while i <= len(p) and flag == 0:
                idx = re.search(uniq_folder_id[count], p[i])
                if idx:
                    curated_foci_name[count] = input_params['csv_folder'] + p[i]
                    flag = 1
                i = i + 1

    # Thresholds for max and min areas of FISH foci per stack
    FISH_min_area_threshold = input_params['FISH_area_min_threshold']
    FISH_max_area_threshold = input_params['FISH_area_max_threshold']

    foci_ac = 1
    image_data_set = list()

    for k in range(len(IF_channel)):

        count = 1

        # read in multipage TIF; 16-bit gray-scale
        IF_image = io.imread(filename)
        FISH_image = io.imread(filename_FISH)

        n_images = IF_image.shape[0]

        # initiate storage variables
        stats_relevant = np.empty(shape=(n_images, 5))
        X_store = np.empty(shape=IF_image.shape())
        XF_store = np.empty(shape=FISH_image.shape())

        image_store = np.empty(shape=X_store.shape())
        fish_store = np.empty(shape=XF_store.shape())

        filename = IF_channel[k]
        filename_FISH = FISH_channel[k]

        for stack in range(n_images):
            X_store[stack, :, :] = IF_image[stack, :, :]

            XF_store[stack, :, :] = FISH_image[stack, :, :]

            mean_intensity = np.mean(XF_store[stack, :, :], axis=None)
            bwF = XF_store[stack, :, :] > (input_params['threshold_multiplier'] * mean_intensity/65535)

            bwF = morphology.remove_small_objects(bwF, FISH_min_area_threshold)

            LF = morphology.label(bwF)
            BF = segmentation.find_boundaries(LF)

            stats = measure.regionprops(LF)

            size_box = input_params['size_box']

            if stats:
                for b_rel in range(len(BF)):
                    if FISH_min_area_threshold <= stats[b_rel]['area'] < FISH_max_area_threshold:
                        centroid = stats[b_rel]['centroid']

                        if (math.floor(centroid[0]) - size_box) > 1 \
                            and (math.floor(centroid[0]) + size_box) < XF_store[stack].shape[0] \
                            and (math.floor(centroid[1]) + size_box) < XF_store[stack].shape[0] \
                            and (math.floor(centroid[1]) - size_box) > 1:

                            flag_end_stack_points = False

                            if stack is 0:
                                flag_end_stack_points = True
                            elif stack <= n_images:
                                box_to_get_x = list(range(math.floor(centroid[1]) - size_box, math.floor(centroid[1]) + size_box))
                                box_to_get_y = list(range(math.floor(centroid[0]) - size_box, math.floor(centroid[0]) + size_box))

                                image_store[count, :, :] = X_store[stack, box_to_get_x, box_to_get_y]
                            else:
                                flag_end_stack_points = True

                            if not flag_end_stack_points:
                                box_to_get_x = list(range(math.floor(centroid[1]) - size_box, math.floor(centroid[1]) + size_box))
                                box_to_get_y = list(range(math.floor(centroid[0]) - size_box, math.floor(centroid[0]) + size_box))
                                fish_store[count, :, :] = XF_store[stack, box_to_get_x, box_to_get_y]
                                stats_relevant[count, 0] = centroid[0]
                                stats_relevant[count, 1] = centroid[1]
                                stats_relevant[count, 2] = stack
                                stats_relevant[count, 3] = stats[b_rel]['area']
                                stats_relevant[count, 4] = np.mean(XF_store[stack, stats[b_rel][stats[b_rel]['coords'][0]], stats[b_rel]['coords'][1]])
                                count = count + 1

        if image_store:

            if input_params['auto_call_foci']:
                com = find_foci_centroid(fish_store, image_store, stats_relevant, input_params)


def find_foci_centroid(fish_store, IF_store, stats, input_params):

    z_threshold = 10
    xy_threshold = 10

    x_pixel = input_params['x_pixel']
    z_pixel = input_params['z_pixel']

    foci = 1

    while foci <= stats.shape[0]:
        com = stats[foci, 0:1]
        stack = stats[foci, 3]
        store_to_remove = list()

        for j in range(foci+1, stats.shape[0]):
            dist = math.sqrt(math.pow(sum(com[1, 0:1] - stats[j, 0:1]), 2))

            if dist < xy_threshold and (stats[j, 3] - stack) <= z_threshold
                store_to_remove = store_to_remove.append(j)

        if com[0] > 1000 & com[1] > 1000:
            stats = np.delete(stats, foci, 0)
            fish_store.remove(foci)
            IF_store.remove(foci)
        else:
            idx = store_to_remove.append(foci)
            idx = sorted(idx)
            stats[foci, 0] = sum(stats[idx, 0]) * stats[idx,3] / sum(stats[idx, 3])
            stats[foci, 1] = sum(stats[idx, 1]) * stats[idx,3] / sum(stats[idx, 3])
            stats[foci, 2] = sum(stats[idx, 2]) * stats[idx,3] / sum(stats[idx, 3])
            stats[foci, 4] = sum(stats[idx, 3]) * stats[idx,3] / sum(stats[idx, 3])
            stats[foci, 3] = sum(stats[idx, 3])

            fish_store[foci] = np.mean(fish_store[idx]) #  @Incomplete not sure if this is correct
            image_store[foci] = np.mean(image_store[idx])








