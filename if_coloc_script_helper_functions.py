import os
import re
import numpy as np
import math
from skimage import io, morphology, segmentation, measure
import pandas as pd
import pickle
from random import shuffle


def generate_output(input_params):

    # function (called slices_of_relevant_images in Krishna's code) that reads in FISH and
    #  IF files to generate output centered on various FISH foci and saves all the capture
    #  foci, as well as the IF information in all stacks for any given FISH foci

    # get all sub-directories where images are stored
    dir_name = input_params['dir_name']
    output_dir_name = input_params['output_dir_name']

    parent_dir = os.path.dirname(dir_name)  # this is the parent directory that holds the "Images" directory

    p = os.listdir(dir_name)
    p = [s for s in p if re.search('E', s) and os.path.isdir(s)]  # must be directories that have an 'E' in the name

    x_pixel = input_params['x_pixel']
    z_pixel = input_params['z_pixel']

    # store IF and FISH information for each replicate
    sub_dir = []
    IF_channel = []
    uniq_folder_id = []
    FISH_channel = []

    for count, file in enumerate(p):
        sub_dir.append(os.path.join(dir_name, file))

        if os.path.isdir(sub_dir[count]):
            local_files = os.listdir(sub_dir[count])

            for j in local_files:
                if re.search(j, input_params['IF_channel']):
                    IF_channel.append(sub_dir[count] + j)
                    uniq_folder_id.append(file[-2:])

                elif re.search(j, input_params['FISH_channel']):
                    FISH_channel.append(sub_dir[count] + j)

    # get previous called foci if automated foci was note called
    curated_foci_name = []
    if not input_params['auto_call_foci_flag']:
        p = os.listdir(input_params['csv_folder'])
        for count in range(len(uniq_folder_id)):
            i = 2
            flag = False
            while i <= len(p) and flag is False:
                idx = re.search(uniq_folder_id[count], p[i])
                if idx:
                    curated_foci_name.append(input_params['csv_folder'] + p[i])
                    flag = True
                i = i + 1

    # Thresholds for max and min areas of FISH foci per stack
    FISH_min_area_threshold = input_params['FISH_area_min_threshold']
    FISH_max_area_threshold = input_params['FISH_area_max_threshold']

    foci_ac = 1
    image_data_set = []
    FISH_IRF = []
    IF_IRF = []
    COM_data = []
    fish_data = []
    IF_data = []

    for k in range(len(IF_channel)):

        count = 0

        # read in multipage TIF; 16-bit gray-scale
        filename = IF_channel[k]
        filename_FISH = FISH_channel[k]

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

                # @Incomplete curate called foci would go here

            else:
                called_foci = pd.read_csv(curated_foci_name) # @Incomplete add ability to load previous called foci
                com = []

            size_box_xy = input_params['size_box']
            size_box_z = math.floor(size_box_xy * x_pixel / z_pixel)


            for foci in range(com.shape[0]):
                centroid = com[foci, :]
                stack_centroid = centroid[2]

                if (stack_centroid - size_box_z) > 1 and (stack_centroid + size_box_z) < n_images:
                    if (math.floor(centroid[0]) - size_box) > 1 \
                        and (math.floor(centroid[0]) + size_box) < XF_store[0, :, :].shape[0] \
                        and (math.floor(centroid[1]) + size_box) < XF_store[0, :, :].shape[0] \
                        and (math.floor(centroid[1]) - size_box) > 1:

                        fish_data[foci_ac] = np.empty(shape=(int(2*size_box_z + 1), size_box, size_box))
                        IF_data[foci_ac] = np.empty(shape=(int(2 * size_box_z + 1), size_box, size_box))
                        for stack in range(int(2*size_box_z + 1)):
                            r_idx = list(range((math.floor(centroid[1]) - size_box), (math.floor(centroid[1]) + size_box)))
                            c_idx = list(range((math.floor(centroid[0]) - size_box), (math.floor(centroid(0)) + size_box)))
                            fish_data[foci_ac][stack, :, :] = XF_store[
                                (round(stack_centroid) - size_box_z - 1 + stack), r_idx, c_idx]
                            IF_data[foci_ac][stack, :, :] = X_store[
                                (round(stack_centroid) - size_box_z - 1 + stack), r_idx, c_idx]

                        FISH_IRF[foci_ac] = convert_3D_image_IRF(fish_data[foci_ac], input_params)
                        IF_IRF[foci_ac] = convert_3D_image_IRF(IF_data[foci_ac], input_params)
                        image_data_set[foci_ac] = filename
                        COM_data[foci_ac] = com[foci, :]
                        foci_ac = foci_ac + 1

    # save output data
    if fish_data:
        save_path = os.path.join(parent_dir, output_dir_name)

        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        output = [FISH_IRF, IF_IRF, fish_data, IF_data, COM_data, image_data_set]

        pickle.dump(output, save_path)



def find_foci_centroid(fish_store, image_store, stats, input_params):

    z_threshold = 10
    xy_threshold = 10

    x_pixel = input_params['x_pixel']
    z_pixel = input_params['z_pixel']

    foci = 1

    while foci <= stats.shape[0]:
        com = stats[foci, 0:1]
        stack = stats[foci, 3]
        store_to_remove = []

        for j in range(foci+1, stats.shape[0]):
            dist = math.sqrt(math.pow(sum(com[1, 0:1] - stats[j, 0:1]), 2))

            if dist < xy_threshold and (stats[j, 3] - stack) <= z_threshold:
                store_to_remove = store_to_remove.append(j)

        if com[0] > 1000 & com[1] > 1000:
            stats = np.delete(stats, foci, axis=0)
            fish_store = np.delete(fish_store, foci, axis=0)
            image_store = np.delete(image_store, foci, axis=0)
        else:
            idx = store_to_remove.append(foci)
            idx = sorted(idx)
            stats[foci, 0] = sum(stats[idx, 0]) * stats[idx, 3] / sum(stats[idx, 3])
            stats[foci, 1] = sum(stats[idx, 1]) * stats[idx, 3] / sum(stats[idx, 3])
            stats[foci, 2] = sum(stats[idx, 2]) * stats[idx, 3] / sum(stats[idx, 3])
            stats[foci, 4] = sum(stats[idx, 3]) * stats[idx, 3] / sum(stats[idx, 3])
            stats[foci, 3] = sum(stats[idx, 3])

            fish_store[foci, :, :] = np.mean(fish_store[idx, :, :], axis=0)
            image_store[foci, :, :] = np.mean(image_store[idx, :, :], axis=0)

            stats = np.delete(stats, store_to_remove, axis=0)
            fish_store = np.delete(fish_store, store_to_remove, axis=0)
            image_store = np.delete(image_store, store_to_remove, axis=0)
            foci = foci + 1

    stats = np.append(stats, stats[:, 0] * x_pixel, axis=1)
    stats = np.append(stats, stats[:, 1] * x_pixel, axis=1)
    stats = np.append(stats, stats[:, 2] * z_pixel, axis=1)
    stats = np.append(stats, stats[:, 3] * x_pixel * x_pixel * z_pixel, axis=1)

    volume_threshold = input_params['volume_threshold']

    foci = 1

    while foci <= stats.shape[0]:
        vol_cur = stats[foci, 8]

        if vol_cur < volume_threshold:
            stats = np.delete(stats, foci, axis=0)
            fish_store = np.delete(fish_store, foci, axis=0)
            image_store = np.delete(image_store, foci, axis=0)
            foci = foci - 1

        foci = foci + 1

    output = np.empty(shape=(stats.shape[0], 4))
    output[:, 0:2] = stats[:, 0:2]
    output[:, 3] = stats[:, 8]

    return output


def convert_3D_image_IRF(data, input_params):

    x_size = data.shape[0]
    y_size = data.shape[1]
    z_size = data.shape[2]

    center_point = [math.ceil(x_size/2), math.ceil(y_size/2), math.ceil(z_size/2)]

    count = 0

    x_pixel = input_params['x_pixel']
    z_pixel = input_params['z_pixel']

    dist_center = np.zeros(shape=(x_size * y_size * z_size, 1))
    intensity = np.zeros(shape=(x_size * y_size * z_size, 1))

    for i in range(x_size):
        for j in range(y_size):
            for k in range(z_size):
                dist_center[count] = math.sqrt((x_pixel * x_pixel * (math.pow(i - center_point[0], 2))) +
                                               (math.pow(j - center_point[1], 2)) +
                                               (z_pixel * z_pixel * (math.pow(k - center_point[2], 2))))

                intensity[count] = float(data[k, i, j])  # I store the stack z in the first dimension
                count = count + 1

    sort_dist_center = sorted(dist_center)
    sort_index = np.argsort(dist_center)

    sort_intensity = intensity[sort_index]

    unique_dist = np.unique(sort_dist_center)

    intensity_output = np.zeros(shape=(unique_dist.shape[0], 1))
    for n in range(len(unique_dist)):
        intensity_output[n] = np.mean(sort_intensity[sort_dist_center == unique_dist[n]])

    output = [sort_dist_center, sort_intensity]

    return output


def auto_call_random_foci(input_params):
    dir_name = input_params['dir_name']
    random_foci_dir = input_params['random_foci_dir']

    p = os.listdir(dir_name)
    p = [s for s in p if re.search('E', s) and os.path.isdir(s)]  # must be directories that have an 'E' in the name

    sub_dir = []
    DNA_channel = []
    uniq_folder_id = []
    output_DNA_file = []

    for count, file in enumerate(p):
        sub_dir.append(os.path.join(dir_name, file))

        if os.path.isdir(sub_dir[count]):
            uniq_folder_id.append(file[-2:])
            local_files = os.listdir(sub_dir[count])

            for j in local_files:
                if re.search(j, input_params['405']):
                    DNA_channel.append(sub_dir[count] + j)
                    output_DNA_file.append(random_foci_dir + 'auto_called_random_foci_' + uniq_folder_id[count])

            cell_classifier_DNA(DNA_channel[count], output_DNA_file[count], 40)


def cell_classifier_DNA(filename, output_file, n_rand):
    X = io.imread(filename)
    n_images = X.shape[0]
    p = []
    mean_data = []
    XDNA_store = np.empty(shape=(n_images, X.shape[1], X.shape[2]))

    for stack in range(n_images):
        XDNA_store[stack, :, :] = X[stack, :, :]

        p.append(np.reshape(XDNA_store[stack, :, :], newshape=(math.pow(XDNA_store[stack, :, :].shape[1],2), 1)))

        mean_data.append(np.mean(p[stack]))

    stack_to_check = mean_data > (0.1*(np.max(mean_data) - np.min(mean_data) + np.min(mean_data)))
    stack_to_check = np.where(stack_to_check)[0]

    threshold = 0.25
    list_of_points = []
    rand_store = np.empty(shape=(5*len(stack_to_check), 4))
    count = 1

    for stack in range(len(stack_to_check)):
        thresh = (np.mean(p[stack_to_check[stack]]) + threshold*np.std(p[stack_to_check[stack]]))/65535
        bw = XDNA_store[stack_to_check[stack], :, :] > thresh

        bw = morphology.remove_small_objects(bw, 100)

        LF = morphology.label(bw)
        BF = segmentation.find_boundaries(LF)

        stats = measure.regionprops(LF)

        k_rel = []

        for k in range(len(BF)):
            boundary = BF[k]

            if len(boundary) > 200:
                list_of_points.append(stats[k]['coords'])
                k_rel.append(k)

        random_points = shuffle(list_of_points)
        random_points = random_points[0:4]  # get 5 random points

        # @Incomplete I don't think this is correct. I don't know how to get all the first terms of a tuple.
        rand_store[count:count+4, 0] = count:count + 4
        rand_store[count:count+4, 1] = random_points[1]
        rand_store[count:count+4, 2] = random_poitns[2]








