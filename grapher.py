import methods

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from skimage import exposure, color


def make_output_images(data, nuclear_mask, image_a, image_b, image_a_bsub, image_b_bsub,
                       input_params, channel_a_name, channel_b_name):
    master_alpha = 0.5
    master_bg_label = 0
    master_font_dict = {'fontsize':8}

    #fig, ax = plt.subplots(nrows=2, ncols=6, figsize=(10, 7.5))
    # ax[0] = nucleus
    # ax[1] = coloc heatmap
    # ax[2] = IF_A original
    # ax[3] = IF_A threshold
    # ax[4] = IF_B original
    # ax[5] = IF_B threshodl
    # gs = gridspec.GridSpec(2, 6)
    #
    # ax = [plt.subplot(gs[:, 0:2]),
    #       plt.subplot(gs[:, -2:]),
    #       plt.subplot(gs[0, 2]),
    #       plt.subplot(gs[0, -2]),
    #       plt.subplot(gs[1, 2]),
    #       plt.subplot(gs[1, -2])]
    #
    # gs.update(wspace=0.01, hspace=0.01)

    # this will make the nucleus image and the heatmap image 2 square axes big
    # gs_nuc = ax[0, 0].get_gridspec()  # @Deprecated
    # gs_heat = ax[0, 3].get_gridspec() # @Deprecated
    # remove the underlying axes
    # for a in ax[0:, 0]:
    #     a.remove()
    # for a in ax[0:, 3]:
    #     a.remove()
    # ax_nuc = fig.add_subplot(gs_nuc[0:, 0])
    # ax_heat = fig.add_subplot(gs_heat[0:, 3])

    fig, ax = plt.subplots(2, 3, figsize=(10, 7.5))

    # plot nucleus image with mask
    nucleus_image = exposure.equalize_adapthist(methods.max_project(data.nucleus_image))
    nuclear_overlay = np.sum(nuclear_mask, axis=0)
    nuclear_overlay[nuclear_overlay > 0] = 1
    nucleus_label_overlay = color.label2rgb(nuclear_overlay, image=nucleus_image,
                                            alpha=master_alpha, image_alpha=1, bg_label=master_bg_label)
    ax[0, 2].imshow(nucleus_label_overlay)
    ax[0, 2].set_title('nucleus mask', master_font_dict)

    # plot IF channels
    image_a = methods.max_project(image_a)
    image_a_thresh = image_a_bsub
    image_a_thresh[np.invert(nuclear_mask)] = 0
    image_a_thresh = methods.max_project(image_a_thresh)

    image_b = methods.max_project(image_b)
    image_b_thresh = image_b_bsub
    image_b_thresh[np.invert(nuclear_mask)] = 0
    image_b_thresh = methods.max_project(image_b_thresh)

    ax[0, 0].imshow(rescale_image(image_a))
    ax[0, 0].set_title('Orig_IF_channel_' + str(channel_a_name), master_font_dict)

    ax[0, 1].imshow(rescale_image(image_a_thresh))
    ax[0, 1].set_title('Thresh_IF_channel_' + str(channel_a_name), master_font_dict)

    ax[1, 0].imshow(rescale_image(image_b))
    ax[1, 0].set_title('Orig_IF_channel_' + str(channel_b_name), master_font_dict)

    ax[1, 1].imshow(rescale_image(image_b_thresh))
    ax[1, 1].set_title('Thresh_IF_channel_' + str(channel_b_name), master_font_dict)

    # multiply images and generate heatmap
    heatmap_image = multiply_images(image_a_thresh, image_b_thresh)

    ax[1, 2].imshow(heatmap_image, cmap='magma')
    ax[1, 2].set_title('Co-localization', master_font_dict)

    # clear axis ticks
    for a in ax.flatten():
        clear_axis_ticks(a)
        a.set_aspect('equal', 'box')

    plt.suptitle(data.sample_name, fontsize=10)
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    plt.tight_layout()

    plt.savefig(os.path.join(data.output_directories['individual_images'],
                             data.sample_name + '_channel_' + str(channel_a_name) +
                             '_channel_' + str(channel_b_name) + '_segmentation.png'), dpi=300)
    plt.savefig(os.path.join(data.output_directories['individual_images'],
                             data.sample_name + '_channel_' + str(channel_a_name) +
                             '_channel_' + str(channel_b_name) + '_segmentation.pdf'))
    plt.close()


def rescale_image(image):
    p2, p98 = np.percentile(image, (2, 98))

    image_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))

    return image_rescale


def multiply_images(image_a, image_b):
    # assuming that image_a and image_b are floats

    combined_image = np.multiply(image_a, image_b)

    return combined_image


def clear_axis_ticks(ax):
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])