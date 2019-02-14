import grapher
import argparse
import os
import sys
import imageio as io

from matplotlib import pyplot as plt
from scipy import ndimage as nd
from scipy import stats as st
import numpy as np
import pandas as pd
from skimage import img_as_float, img_as_uint, morphology, measure, color
from sklearn.cluster import KMeans
from datetime import datetime
import math
from itertools import compress
from types import SimpleNamespace
import json


def parse_arguments(parser):

    # required arguments
    parser.add_argument("parent_dir")
    parser.add_argument("output_path", type=str)

    # optional arguments
    parser.add_argument("--tm", type=float, default=1.5)  # for background subtraction. Background + std*tm

    # flags
    # parser.add_argument("--manual", dest="autocall_flag", action="store_false", default=True)

    input_params = parser.parse_args()

    return input_params


def make_output_directories(input_params):
    output_parent_dir = input_params.output_path

    output_dirs = {'parent': output_parent_dir,
                   'individual_images': os.path.join(output_parent_dir, 'individual_images')}

    # make folders if they don't exist
    if not os.path.isdir(output_parent_dir):
        os.mkdir(output_parent_dir)

    for key, folder in output_dirs.items():
        if key is not 'output_parent':
            if not os.path.isdir(folder):
                if not os.path.isdir(os.path.dirname(folder)):  # so I guess .items() is random order of dictionary keys. So when making subfolders, if the parent doesn't exist, then we would get an error. This accounts for that.
                    os.mkdir(os.path.dirname(folder))

                os.mkdir(folder)

    return output_dirs


def load_images(replicate_files, input_params, parent_dir):
    data = SimpleNamespace()  # this is the session data object that will be passed to functions

    # get replicate sample name
    nd_file_name = [n for n in replicate_files if '.nd' in n]
    if len(nd_file_name) == 1:
        sample_name = get_sample_name(nd_file_name[0])
        data.sample_name = sample_name
    else:
        print('Error: Found too many .nd files in sample directory')
        sys.exit(0)

    print(sample_name)

    # load images
    nucleus_image_file = [f for f in replicate_files if all(['405 DAPI' in f, get_file_extension(f) == '.TIF'])]
    if len(nucleus_image_file) < 1:
        print('Error: Could not find nucleus image file')
        sys.exit(0)

    nucleus_image_path = os.path.join(input_params.parent_dir, parent_dir, nucleus_image_file[0])
    nucleus_image = io.volread(nucleus_image_path)  # image is [z, x, y] array
    data.nucleus_image = nucleus_image

    protein_image_files = [p for p in replicate_files if
                           all(['405 DAPI' not in p,
                                get_file_extension(p) == '.TIF'])]
    if len(protein_image_files) < 1:
        print('Error: Could not find protein image files')
        sys.exit(0)

    protein_image_paths = []
    protein_images = []
    protein_channel_names = []
    for idx, p in enumerate(protein_image_files):
        protein_image_paths.append(os.path.join(input_params.parent_dir, parent_dir, p))
        protein_channel_names.append(find_image_channel_name(p))
        protein_images.append(io.volread(protein_image_paths[idx]))

    data.protein_images = protein_images
    data.protein_channel_names = protein_channel_names

    data.multi_flag = False  # this tells us if we have three channels to compare
    if len(data.protein_images) > 2:
        data.multi_flag = True

    return data


def analyze_replicate(data, input_params, individual_replicate_output, channel_a_idx=0, channel_b_idx=1):

    # get nuclear mask
    nuclear_mask = find_nucleus(data.nucleus_image, input_params)
    data.nuclear_mask = nuclear_mask

    # find and subtract backgrounds of IF channels
    image_a = data.protein_images[channel_a_idx]
    image_a = img_as_float(image_a)

    image_b = data.protein_images[channel_b_idx]
    image_b = img_as_float(image_b)

    data.image_a_orig = image_a
    data.image_b_orig = image_b
    channel_a = data.protein_channel_names[channel_a_idx]
    channel_b = data.protein_channel_names[channel_b_idx]

    image_a_bsub, threshold_a = subtract_background(image_a, input_params)
    image_b_bsub, threshold_b = subtract_background(image_b, input_params)

    # data.image_a_bsub = image_a_bsub
    # data.image_b_bsub = image_b_bsub  # IMPORTANT: This gets over-written with multichannel comparisons

    # filter on nuclear pixels
    image_a_bsub_filt = image_a_bsub[nuclear_mask]
    image_b_bsub_filt = image_b_bsub[nuclear_mask]

    # make it 1D for statistics
    image_a_1D = image_a_bsub_filt.reshape((-1, 1))
    image_b_1D = image_b_bsub_filt.reshape((-1, 1))

    # Pearson correlation
    p_rho, p_pval = st.pearsonr(image_a_1D, image_b_1D)
    p_rho = p_rho[0]  # because these are numpy arrays and we just want the value
    p_pval = p_pval[0]

    # Spearman correlation
    s_rho, s_pval = st.spearmanr(image_a_1D, image_b_1D)

    # Manders coefficients
    image_a_mask = np.full(image_a_1D.shape, fill_value=False, dtype=bool)
    image_a_mask[np.where(image_a_1D > 0)] = True

    image_b_mask = np.full(image_b_1D.shape, fill_value=False, dtype=bool)
    image_b_mask[np.where(image_b_1D > 0)] = True

    m_rho_a = find_manders_coeff(image_a_1D, image_b_mask)  # m_a = M1, m_b = M2 for Manders coefficients
    m_rho_b = find_manders_coeff(image_b_1D, image_a_mask)

    individual_replicate_output = individual_replicate_output.append({'sample'         : data.sample_name,
                                                                      'replicate'      : input_params.replicate_count_idx,
                                                                      'channel_a'      : int(channel_a),
                                                                      'channel_b'      : int(channel_b),
                                                                      'pearson_rho'    : p_rho,
                                                                      'pearson_p-val'  : p_pval,
                                                                      'spearman_rho'   : s_rho,
                                                                      'spearman_p-val' : s_pval,
                                                                      'manders_1'      : m_rho_a,
                                                                      'manders_2'      : m_rho_b},
                                                                     ignore_index=True)

    # generate output images and save them
    grapher.make_output_images(data, nuclear_mask, image_a, image_b, image_a_bsub, image_b_bsub,
                               input_params, channel_a, channel_b)

    return individual_replicate_output, data




def find_nucleus(image, input_params):
    # for co_IF analysis, we will do a blurring, then maximum intensity projection, then clustering of pixels into
    # 2 clusters with k-means. From this, we will then cluster the entire image/z-stack using this clustering solution

    image = nd.gaussian_filter(image, sigma=2.0)
    image_sample = max_project(image)  # max project the image because we just care about finding nuclei
    image_sample = img_as_float(image_sample)
    image = img_as_float(image)
    # threshold_multiplier = 0.25

    # trial method. Use K-means clustering on image to get nuclear pixels
    image_sample_1d = image_sample.reshape((-1, 1))
    #
    cluster_solution = KMeans(n_clusters=2, random_state=0).fit(image_sample_1d)
    clusters = cluster_solution.predict(image_sample_1d)
    #
    cluster_mean = []
    for c in range(2):
        cluster_mean.append(np.mean(image_sample_1d[clusters == c]))

    nuclear_cluster = np.argmax(cluster_mean)

    # now use clustering solution on the entire z-stack image
    image_1d = image.reshape((-1, 1))

    clusters_all = cluster_solution.predict(image_1d)

    clusters_all = np.reshape(clusters_all, newshape=image.shape)
    nuclear_mask = np.full(shape=image.shape, fill_value=False, dtype=bool)
    nuclear_mask[clusters_all == nuclear_cluster] = True

    # # # simple thresholding
    # mean_intensity = np.mean(image_sample_1d)
    # std_intensity = np.std(image_sample_1d)
    #
    # threshold = mean_intensity + (std_intensity * 0.8)
    # nuclear_mask = np.full(shape=image.shape, fill_value=False, dtype=bool)
    # nuclear_mask[np.where(image > threshold)] = True

    nuclear_binary = nd.morphology.binary_dilation(nuclear_mask)

    nuclear_binary = nd.morphology.binary_fill_holes(nuclear_mask)

    nuclear_binary = nd.binary_erosion(nuclear_binary)  # to try to get rid of touching nuclei. Need to do better!


    # nuclear_binary_labeled, num_of_regions = nd.label(nuclear_binary)

    # nuclear_regions = measure.regionprops(nuclear_binary_labeled)
    # nuclear_regions = nd.find_objects(nuclear_binary_labeled) # @Deprecated

    return nuclear_mask


def max_project(image):
    projection = np.max(image, axis=0)

    return projection


def get_file_extension(file_path):
    file_ext = os.path.splitext(file_path)

    return file_ext[1]  # because splitext returns a tuple, and the extension is the second element


def find_image_channel_name(file_name):
    str_idx = file_name.find('Conf ')  # this is specific to our microscopes file name format
    channel_name = file_name[str_idx + 5 : str_idx + 8]

    return channel_name


def get_sample_name(nd_file_name):
    sample_name, ext = os.path.splitext(nd_file_name)

    return sample_name


def adjust_excel_column_width(writer, output):
    for key, sheet in writer.sheets.items():
        for idx, name in enumerate(output.columns):
            col_width = len(name) + 2
            sheet.set_column(idx, idx, col_width)

    return writer



def write_output_params(input_params):

    # write parameters that were used for this analysis
    # output_params = {'parent_dir': input_args.parent_dir,
    #                  'output_path' : input_args.output_path,
    #                  'fish_channel' : input_args.fish_channel,
    #                  'time_of_analysis': datetime.now(),
    #                  'tm': input_args.tm,
    #                  'min_a': input_args.min_a,
    #                  'max_a': input_args.max_a,
    #                  'c': input_args.c,
    #                  'b': input_args.b,
    #                  'auto_call_flag' : input_args.autocall_flag
    #                  }

    with open(os.path.join(input_params.output_path, 'output_analysis_parameters.txt'), 'w') as file:
        file.write(json.dumps(input_params, default=str))


def subtract_background(input_image, input_params):
    image_hist, image_bin_edges = np.histogram(input_image, bins='auto')

    max_image = max_project(input_image)
    image_std = np.std(max_image)

    threshold_multiplier = input_params.tm

    background_threshold = image_bin_edges[np.argmax(image_hist)]  # assumes that the max hist peak corresponds to background pixels
    background_threshold = background_threshold + (image_std * threshold_multiplier)

    output_image = input_image - background_threshold
    output_image[output_image < 0.0] = 0.0

    # output_image = np.reshape(output_image, input_image.shape)
    return output_image, background_threshold


def find_manders_coeff(image, mask):
    # image is a 1D array of the pixels
    # mask is a 1D boolean array showing pixels where the other channel > 0

    r_co = np.sum(image[mask])
    r_total = np.sum(image)

    coeff = r_co/r_total

    return coeff