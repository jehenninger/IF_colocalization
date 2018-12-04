# this script is to identify lacO spots and quantify fluorescence of another channel in them
import argparse
import sys
import os
import math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from skimage import io, filters, measure, color, exposure, morphology, feature, img_as_float, img_as_uint
from scipy import ndimage as ndi
from itertools import compress
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from datetime import datetime
import warnings
import json
# import progressbar

# @TODO Add ability to load in previous analysis and add new files?
# @TODO Make function to re-make boxplot with loaded output excel file
# @TODO When nuclear stain is not available, use the lacO channel and remove the brightest pixels
# @TODO quantify DAPI signal at tether to tell if chromatin is closed
# @TODO Output stats on ALL the tether spots that it finds, and which ones were kept. Would make it easier to identify
#       which optional parameters to change

# parse input
parser = argparse.ArgumentParser()
parser.add_argument("metadata_file", help="metadata_file is the Excel file with paths to images with info")

parser.add_argument("--o", type=str)
parser.add_argument("--ni", type=float)
parser.add_argument("--na", type=float)
parser.add_argument("--ti", type=float)
parser.add_argument("--tamin", type=float)
parser.add_argument("--tamax", type=float)
parser.add_argument("--tc", type=float)

input_args = parser.parse_args()

circ = lambda r: (4 * math.pi * r.area) / (r.perimeter * r.perimeter)

metadata_path = input_args.metadata_file

if not os.path.isfile(metadata_path):
    sys.exit('ERROR: Could not find metadata file')
    
def read_metadata(input_args):
    # read in images and metadata

    # metadata_path = "/Users/jon/data_analysis/young/to_do/181118_lacO/metadata.xlsx"

    metadata_dir = os.path.dirname(input_args.metadata_file)
    metadata = pd.read_excel(metadata_path, sheet_name=0, header=0)
    metadata_name = os.path.basename(input_args.metadata_file)
    metadata_name = os.path.splitext(metadata_name)[0]
    
    output_dirs = []

    if input_args.o:
        output_dirs.append(os.path.join(metadata_dir, input_args.o))
    else:
        output_dirs.append(os.path.join(metadata_dir, metadata_name + '_output'))

    output_dirs.append(os.path.join(output_dirs[0], 'eps_files'))
    output_dirs.append(os.path.join(output_dirs[0], 'png_files'))

    if not os.path.isdir(output_dirs[0]):
        os.mkdir(output_dirs[0])
    if not os.path.isdir(output_dirs[1]):
        os.mkdir(output_dirs[1])
    if not os.path.isdir(output_dirs[2]):
        os.mkdir(output_dirs[2])


    return metadata, output_dirs

def run_analysis(metadata, output_dirs, input_args):
    if input_args.ni:
        nuclear_intensity_threshold_multiplier = input_args.ni
    else:
        nuclear_intensity_threshold_multiplier = 2

    if input_args.na:
        nuclear_min_area_threshold = input_args.na
    else:
        nuclear_min_area_threshold = 10000

    if input_args.ti:
        tether_intensity_threshold_multiplier = input_args.ti
    else:
        tether_intensity_threshold_multiplier = 1.5

    if input_args.tamin:
        tether_min_area_threshold = input_args.tamin
    else:
        tether_min_area_threshold = 150

    if input_args.tamax:
        tether_max_area_threshold = input_args.tamax
    else:
        tether_max_area_threshold = 2000

    if input_args.tc:
        tether_circ_threshold = input_args.tc
    else:
        tether_circ_threshold = 0.8
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # get number of samples
        samples = np.unique(metadata['sample_name'])
        num_of_samples = len(samples)

        # initialize output dataframe structure
        individual_output = pd.DataFrame(columns = ['sample','replicate', 'tether_id', 'nucleus_id', 'mean_intensity_in', 'mean_intensity_out', 'norm_enrichment'])

        for s in samples:
            print()
            print("Sample: ", s)
            print()
            metadata_sample = metadata[(metadata['sample_name'] == s)].copy()

            # get number of replicates
            replicates = np.unique(metadata_sample['replicate'])
            num_of_replicates = len(replicates)

            # progress_iterations = num_of_replicates

            #with progressbar.ProgressBar(max_value=progress_iterations) as bar:
            #### loop through replicates ####
                # bar_count = 0
            for r in replicates:
                # print('replicate: ', r)
                output_fig, output_ax = plt.subplots(nrows=2, ncols=3)

                metadata_replicate = metadata_sample[metadata_sample['replicate'] == r].copy()

                file_paths = metadata_replicate['sample_path'].copy()
                channel_names = metadata_replicate['channel_name'].copy()

                #### parse metadata to find specific channels ####
                tether_idx = np.where(metadata_replicate['channel_type'] == 'tether')[0]
                IF_a_idx = np.where(metadata_replicate['channel_type'] == 'IF_a')[0]
                IF_b_idx = np.where(metadata_replicate['channel_type'] == 'IF_b')[0]
                nuclear_idx = np.where(metadata_replicate['channel_type'] == 'nuclear')[0]

                #### read images of each channel type ####
                if len(tether_idx) > 0:  # this is annoying to do it this way, but Python interprets a 0 index as "False"
                    tether_I = io.imread(file_paths.iloc[tether_idx[0]])

                    # test if multipage tiff and do max intensity if so
                    num_dims = len(tether_I.shape)
                    if num_dims == 3:
                        tether_I = max_z_projection(tether_I)
                    elif num_dims == 2:
                        pass
                    else:
                        sys.exit('ERROR: Could not interpret TIF file')

                    tether_name = channel_names.iloc[tether_idx[0]]
                    output_ax[0, 1].imshow(tether_I, cmap='gray')
                    output_ax[0, 1].set_title(tether_name)
                    output_ax[0, 1].get_xaxis().set_ticks([])
                    output_ax[0, 1].get_yaxis().set_ticks([])
                else:
                    sys.exit('ERROR: Could not find the tether channel image')

                if len(IF_a_idx) > 0:
                    IF_a_I = io.imread(file_paths.iloc[IF_a_idx[0]])

                    # test if multipage tiff and do max intensity if so
                    num_dims = len(IF_a_I.shape)
                    if num_dims == 3:
                        IF_a_I = max_z_projection(IF_a_I)
                    elif num_dims == 2:
                        pass
                    else:
                        sys.exit('ERROR: Could not interpret TIF file')

                    IF_a_name = channel_names.iloc[IF_a_idx[0]]
                    output_ax[0, 2].imshow(IF_a_I, cmap='gray')
                    output_ax[0, 2].set_title(IF_a_name)
                    output_ax[0, 2].get_xaxis().set_ticks([])
                    output_ax[0, 2].get_yaxis().set_ticks([])

                if len(IF_b_idx) > 0:
                    IF_b_I = io.imread(file_paths.iloc[IF_b_idx[0]])

                    # test if multipage tiff and do max intensity if so
                    num_dims = len(IF_b_I.shape)
                    if num_dims == 3:
                        IF_b_I = max_z_projection(IF_b_I)
                    elif num_dims == 2:
                        pass
                    else:
                        sys.exit('ERROR: Could not interpret TIF file')

                    IF_b_name = channel_names.iloc[IF_b_idx[0]]
                    output_ax[1, 2].imshow(IF_b_I, cmap='gray')
                    output_ax[1, 2].set_title(IF_b_name)
                    output_ax[1, 2].get_xaxis().set_ticks([])
                    output_ax[1, 2].get_yaxis().set_ticks([])

                else:
                    sys.exit('ERROR: Could not find the IF channel image')

                nuclear_stain = False
                if len(nuclear_idx) > 0:
                    nuclear_stain = True
                    nuclear_I = io.imread(file_paths.iloc[nuclear_idx[0]])

                    # test if multipage tiff and do max intensity if so
                    num_dims = len(nuclear_I.shape)
                    if num_dims == 3:
                        nuclear_I = max_z_projection(nuclear_I)
                    elif num_dims == 2:
                        pass
                    else:
                        sys.exit('ERROR: Could not interpret TIF file')

                    nuclear_name = channel_names.iloc[nuclear_idx[0]]
                    output_ax[0, 0].imshow(nuclear_I, cmap='gray')
                    output_ax[0, 0].set_title(nuclear_name)
                    output_ax[0, 0].get_xaxis().set_ticks([])
                    output_ax[0, 0].get_yaxis().set_ticks([])


                #### identify nuclei and make mask ####
                if nuclear_stain:
                    nuclear_I = filters.gaussian(nuclear_I, sigma=2)
                    # test_fig, test_ax = plt.subplots(nrows=1, ncols=2)
                    # test_ax[0].imshow(nuclear_I)

                    # use k-means clustering to identify nuclear pixels on a sample of the image. Find mean of 
                    # these pixels to automatically set threshold multiplier
                    # image_sample = shuffle(nuclear_I, random_state=0)[0:100, 0:100]

                    # image_sample = np.array(nuclear_I, dtype=np.float64) / 65535  # converting to work with KMeans
                    image_sample = img_as_float(nuclear_I)
                    image_sample = image_sample.reshape((-1, 1)) # make 1-D array for indexing later

                    clusters = KMeans(n_clusters=2, random_state=0).fit_predict(image_sample)

                    # clustered_image = np.zeros((nuclear_I.shape[0], nuclear_I.shape[1]))
                    # label_idx = 0
                    # for i in range(nuclear_I.shape[0]):
                        # for j in range(nuclear_I.shape[1]):
                            # clustered_image[i][j] = clusters[label_idx]
                            # label_idx += 1

                    # test_ax[1].imshow(clustered_image)

                    # plt.show()

                    # unique, counts = np.unique(clusters, return_counts=True)

                    clusters = np.reshape(clusters, newshape=(nuclear_I.shape[0], nuclear_I.shape[1]))
                    # percentile_threshold = 5
                    # cluster_percentile = []
                    cluster_mean = []
                    for c in range(2):
                        # cluster_percentile.append(65535 * np.percentile(nuclear_I[clusters == c], percentile_threshold))
                        cluster_mean.append(65535 * np.mean(nuclear_I[clusters == c]))

                    # print(cluster_percentile)
                    # print(cluster_mean)
                    nuclear_cluster = np.argmax(cluster_mean)

                    # nuc_threshold = int(round(np.max(cluster_percentile)))
                    # print("Nuclear threshold is %d" % nuc_threshold)


                    # nuc_threshold = filters.threshold_mean(nuclear_I)

                    # nuclear_binary = nuclear_I > (nuc_threshold * nuclear_intensity_threshold_multiplier)
                    nuclear_binary = clusters == nuclear_cluster  # this just trusts the kmeans to cluster the nuclei. No thresholds.
                    # nuclear_binary = nuclear_I > (nuc_threshold)
                    nuclear_binary = ndi.morphology.binary_fill_holes(nuclear_binary)

                    nuclear_binary = morphology.erosion(nuclear_binary)
                    nuclear_binary = morphology.erosion(nuclear_binary)
                    nuclear_binary = morphology.erosion(nuclear_binary)
                    nuclear_binary = morphology.erosion(nuclear_binary)
                    nuclear_binary = morphology.erosion(nuclear_binary)
                    nuclear_binary = morphology.erosion(nuclear_binary)
                    nuclear_binary = morphology.erosion(nuclear_binary)
                    nuclear_binary = morphology.erosion(nuclear_binary)

                    nuclear_binary_labeled = measure.label(nuclear_binary)

                    nuclear_regionprops = measure.regionprops(nuclear_binary_labeled)

                    nuclear_mask = np.full(shape=(nuclear_I.shape[0], nuclear_I.shape[1]), fill_value=False, dtype=bool)

                    keep_nuclear_region = [False] * len(nuclear_regionprops)
                    for idx, region in enumerate(nuclear_regionprops):
                        if region.area >= nuclear_min_area_threshold:
                            for coords in region.coords:
                                nuclear_mask[coords[0], coords[1]] = True
                            keep_nuclear_region[idx] = True
                        else:
                            label_to_delete = region.label
                            nuclear_binary_labeled[nuclear_binary_labeled == label_to_delete] = 0

                    nuclear_binary_filtered_labeled = measure.label(nuclear_mask)
                    nuclear_filtered_regionprops = measure.regionprops(nuclear_binary_filtered_labeled)

                    # plot filtered nucleus
                    adjusted_nuclear = exposure.equalize_adapthist(nuclear_I)
                    nuclear_overlay = color.label2rgb(nuclear_binary_filtered_labeled, image=adjusted_nuclear,
                                                      alpha=0.5, image_alpha=1, bg_label=0, bg_color=None)

                    output_ax[1, 0].imshow(nuclear_overlay)

                    for idx, region in enumerate(nuclear_filtered_regionprops):
                        nucleus_center = region.centroid
                        output_ax[1, 0].text(int(round(nucleus_center[1])),
                                             int(round(nucleus_center[0])),
                                             idx+1, color='w', fontsize=8)

                    output_ax[1, 0].set_title('Filtered nuclei')
                    output_ax[1, 0].get_xaxis().set_ticks([])
                    output_ax[1, 0].get_yaxis().set_ticks([])
                else:
                    output_ax[0, 0].axis('off')
                    output_ax[1, 0].axis('off')

                #### find lacO spot ####

                # identify tether and make mask
                if len(tether_idx) > 0:
                    tether_I = filters.gaussian(tether_I, sigma=2)
                    tether_threshold = filters.threshold_mean(tether_I)

                    tether_binary = tether_I > (tether_threshold * tether_intensity_threshold_multiplier)
                    tether_binary = ndi.morphology.binary_fill_holes(tether_binary)

                    tether_binary_labeled = measure.label(tether_binary)

                    tether_regionprops = measure.regionprops(tether_binary_labeled)

                    tether_mask = np.full(shape=(tether_I.shape[0], tether_I.shape[1]), fill_value=False, dtype=bool)

                    for idx, region in enumerate(tether_regionprops):
                        # print("Area for %d is %.2f" % (idx, region.area))
                        # print("Circularity for %d is %.2f" % (idx, circ(region)))
                        if (tether_min_area_threshold <= region.area <= tether_max_area_threshold) and (circ(region) >= tether_circ_threshold):
                            # test if tether region centroid is inside nuclear mask (i.e. = True in mask)
                                    if nuclear_stain:
                                        center_x, center_y = region.centroid
                                        center_x = int(round(center_x))
                                        center_y = int(round(center_y))

                                        if nuclear_mask[center_x, center_y]:
                                            for coords in region.coords:
                                                tether_mask[coords[0], coords[1]] = True
                                        else:
                                            label_to_delete = region.label
                                            tether_binary_labeled[tether_binary_labeled == label_to_delete] = 0
                                    else:
                                        for coords in region.coords:  # this is the case of no nuclear stain; just keep everything that meets size threshold
                                            tether_mask[coords[0], coords[1]] = True

                        else:
                            label_to_delete = region.label
                            tether_binary_labeled[tether_binary_labeled == label_to_delete] = 0



                    #### get filtered tether masks ####
                    #tether_mask = ndi.morphology.binary_dilation(tether_mask, iterations=3)  # add 3 dilations to the lacO spot
                    tether_filtered_binary_labeled = measure.label(tether_mask)
                    tether_filtered_regionprops = measure.regionprops(tether_filtered_binary_labeled)

                    # loop through filtered tethers
                    nucleus_label = [-1] * len(tether_filtered_regionprops)
                    tether_intensity = [0] * len(tether_filtered_regionprops)
                    keep_tether_idx = [True] * len(tether_filtered_regionprops)
                    keep_tether_idx = np.asarray(keep_tether_idx)

                    if nuclear_stain:
                        for idx, region in enumerate(tether_filtered_regionprops):
                            bbox = region.bbox
                            # get mean intensity of tether image within bounding box of region
                            tether_intensity[idx] = np.mean(tether_I[bbox[0]:bbox[2], bbox[1]:bbox[3]])

                            # get label ID of nucleus with the individual tether
                            # include_tether = True
                            if nuclear_stain:
                                center_x, center_y = region.centroid
                                center_x = int(round(center_x))
                                center_y = int(round(center_y))
                                nucleus_label[idx] = nuclear_binary_filtered_labeled[center_x, center_y]

                        #print("Tether intensity: ", tether_intensity)
                        for n in np.unique(nucleus_label):
                            # only include max mean intensity tether if there are multiple objects per nucleus
                            multitether_test = np.isin(nucleus_label, n)

                            if sum(multitether_test) > 1:
                                temp_tether_intensity = np.asarray(tether_intensity)
                                temp_tether_intensity[nucleus_label != n] = -1
                                #print("Temp tether intensity: ", temp_tether_intensity)

                                max_tether_idx = np.argmax(temp_tether_intensity)
                                #print("Max tether index: ", max_tether_idx)

                                dim_tether_idx = [i for i, val in enumerate(temp_tether_intensity)
                                                  if (i != max_tether_idx) and (val != -1)]

                                #print("Dim tether IDX: ", dim_tether_idx)

                                keep_tether_idx[dim_tether_idx] = False


                            # if max_tether > tether_intensity[idx]:
                            #   include_tether = False
                            #    for coords in region.coords:  # update filtered tether mask to get rid of this tether
                            #        tether_filtered_binary_labeled[coords[0], coords[1]] = 0

                    #print("Keep tether IDX: ", keep_tether_idx)

                    count = 0
                    for idx, region in enumerate(tether_filtered_regionprops):
                        if keep_tether_idx[idx]:
                            #initialize empty individual tether mask and fill with individual tether
                            individual_tether_mask = np.full(shape=(tether_mask.shape[0], tether_mask.shape[1]), fill_value=False, dtype=bool)

                            for coords in region.coords:
                                individual_tether_mask[coords[0], coords[1]] = True

                            # get intensity in IF channel of the tether with dilation
                            IF_a_mean_intensity_in = np.mean(IF_a_I[individual_tether_mask])
                            IF_b_mean_intensity_in = np.mean(IF_b_I[individual_tether_mask])
                            # print("Mean intensity in spot: ", IF_a_mean_intensity_in)

                            if nuclear_stain:  # get intensity in IF channel for the rest of the nucleus for that tether
                                individual_nuclei_mask = nuclear_binary_filtered_labeled == nucleus_label[idx]
                                individual_nuclei_mask[individual_tether_mask] = False
                                IF_a_mean_intensity_out = np.mean(IF_a_I[individual_nuclei_mask])
                                IF_b_mean_intensity_out = np.mean(IF_b_I[individual_nuclei_mask])
                            else:  # get intensity in IF channel for the whole image minus the tether if no nuclear stain
                                entire_image_without_tether_mask = np.full(shape=(tether_I.shape[0], tether_I.shape[1]),
                                                                           fill_value=True, dtype=bool)
                                entire_image_without_tether_mask[individual_tether_mask] = False
                                IF_a_mean_intensity_out = np.mean(IF_a_I[entire_image_without_tether_mask])
                                IF_b_mean_intensity_out = np.mean(IF_b_I[entire_image_without_tether_mask])

                            # print("Mean intensity outside spot in nucleus: ", IF_a_mean_intensity_out)

                            norm_enrichment_A = IF_a_mean_intensity_in/IF_a_mean_intensity_out
                            norm_enrichment_B = IF_b_mean_intensity_in/IF_b_mean_intensity_out

                            if nuclear_stain:
                                nucleus_id = nucleus_label[idx]
                            else:
                                nucleus_id = 0

                            individual_output = individual_output.append({'sample': s,'replicate': r,
                                                                          'tether_id': count+1,
                                                                          'nucleus_id': nucleus_id,
                                                                          'mean_intensity_in_A': IF_a_mean_intensity_in,
                                                                          'mean_intensity_out_A': IF_a_mean_intensity_out,
                                                                          'mean_intensity_in_B': IF_b_mean_intensity_in,
                                                                          'mean_intensity_out_B': IF_b_mean_intensity_out,
                                                                          'norm_enrichment_A': norm_enrichment_A,
                                                                          'norm_enrichment_B': norm_enrichment_B},
                                                                           ignore_index=True)
                            count = count + 1
                        else:
                            for coords in region.coords:
                                tether_filtered_binary_labeled[coords[0], coords[1]] = 0


                    # plot final tether mask
                    try:
                        adjusted_tether = exposure.equalize_adapthist(tether_I)
                    except:
                        adjusted_tether = tether_I

                    tether_overlay = color.label2rgb(tether_filtered_binary_labeled, image=adjusted_tether,
                                                     alpha=0.5, image_alpha=1, bg_label=0, bg_color=None)

                    tether_filtered_regionprops = list(compress(tether_filtered_regionprops, keep_tether_idx))

                    output_ax[1, 1].imshow(tether_overlay)
                    for idx, region in enumerate(tether_filtered_regionprops):
                        text_offset = 30
                        tether_center = region.centroid
                        output_ax[1, 1].text(int(round(tether_center[1])) + text_offset,
                                             int(round(tether_center[0])) + 3*text_offset, idx+1, color='w', fontsize=8)
                    output_ax[1, 1].set_title('Filtered tether')
                    output_ax[1, 1].get_xaxis().set_ticks([])
                    output_ax[1, 1].get_yaxis().set_ticks([])

                    # make last axis that we don't use empty
                    output_ax[1, 2].axis('off')

                    # save output figures
                    plt.savefig(os.path.join(output_dirs[2], s + '_rep' + str(r) + '_masks.png'), dpi=300, format='png')
                    plt.savefig(os.path.join(output_dirs[1], s + '_rep' + str(r) + '_masks.eps'), format='eps')

                    plt.close()

                    print("Finished replicate ", r, " at ", datetime.now())
                        #bar.update(bar_count)
                        #bar_count += 1

        # save Excel output
        individual_output.to_excel(os.path.join(output_dirs[0], 'output.xlsx'), index=False)
        # print(individual_output)

        print()
        
    #### combine replicates in experiments ####

    # need to implement both channels A and B for this. Currently only does one channel.
    # experimental_groups = np.unique(individual_output['sample'])
    #
    # norm_enrichment = []
    # for e in experimental_groups:
    #     norm_enrichment.append(individual_output[(individual_output['sample'] == e)]['norm_enrichment'])
    #
    # exp_fig, exp_ax = plt.subplots()
    # exp_ax.set_title('Normalized enrichment')
    # exp_ax.boxplot(norm_enrichment, labels=experimental_groups, showfliers=False)
    #
    # for i in range(len(experimental_groups)):
    #     y = norm_enrichment[i]
    #     x = np.random.normal(1+i, 0.04, size=len(y))
    #     exp_ax.plot(x, y, 'b.', markersize=15, markeredgewidth=0, alpha=0.5)
    #
    # for x in exp_ax.get_xticklabels():
    #     x.set_rotation(90)
    #     x.set_fontsize(6)
    #
    # plt.tight_layout()
    #
    # plt.savefig(os.path.join(output_dirs[0], 'experimental_boxplot.png'), dpi=300, format='png')
    # plt.savefig(os.path.join(output_dirs[0], 'experimental_boxplot.eps'), format='eps')

    output_params = {'metadata_file': metadata_path,
                     'time_of_analysis': datetime.now(),
                     'nuclear_min_area_threshold': nuclear_min_area_threshold,
                     'tether_intensity_threshold_multiplier': tether_intensity_threshold_multiplier,
                     'tether_min_area_threshold': tether_min_area_threshold,
                     'tether_max_area_threshold': tether_max_area_threshold,
                     'tether_circ_threshold': tether_circ_threshold}

    with open(os.path.join(output_dirs[0], 'output_analysis_parameters.txt'), 'w') as file:
        file.write(json.dumps(output_params, default=str))

    print("Finished all: ", datetime.now())


def max_z_projection(image):
    image = img_as_float(image)
    # image = np.array(image, dtype=np.float64) / 65535
    max_image = np.max(image, axis=0)
    max_image = img_as_uint(max_image)

    return max_image


######## RUN PROGRAM ########
metadata, output_dirs = read_metadata(input_args)
run_analysis(metadata, output_dirs, input_args)