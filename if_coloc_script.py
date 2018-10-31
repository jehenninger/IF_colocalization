import sys
import os
import pandas as pd
import if_coloc_script_helper_functions as helper


def run_IF_center_average_analysis(metadata_path, FISH_area_thresh=(12, 200), auto_call_foci=True, foci_path='', auto_call_random_foci=True):
    # Directory_name will point to the metadata file that has the following information:
    #
    # 1. image_directory:  folder under which replicate/different image data sets
    # are stored. These image data sets must be stored under folders starting with "E", e.g. E01/, E02/,
    # etc...
    #
    # 2. output_name: name of the output folder that will be generated.
    # Output will go into directory with image data sets
    #
    # 3. FISH_name: name of protein or FISH gene to center on
    #
    # 4. FISH_channel: excitation channel (488 = GFP, 561 = Red, 642 = Far Red) Needs to be present in image file name!
    #
    # 5. IF_name: name of protein to measure around FISH/center
    #
    # 6. IF_channel: excitation channel (488 = GFP, 561 = Red, 642 = Far Red). Needs to be present in image file name!
    #
    # 7. threshold_multi: Threshold multiplier to call foci. Generally FISH is 2. Best to look at
    # example images and do a line histogram to see how much above the mean fluorescence the signal is.

    # load metadata and parse data
    metadata = pd.read_excel(metadata_path, sheet_name=0)

    num_files = len(metadata.index)

    for index, row in metadata.iterrows():
        metadata_params = {'dir_name': row['image_directory'],
                           'output_dir_name': row['output_name'],
                           'FISH_name': row['FISH_name'],
                           'FISH_channel': row['FISH_channel'],
                           'IF_name': row['IF_name'],
                           'IF_channel': row['IF_channel'],
                           'threshold_multiplier': row['threshold_multiplier']}

        input_params = metadata_params

        # area threshold for foci (pixel^2)
        input_params['FISH_area_min_threshold'] = FISH_area_thresh[0]
        input_params['FISH_area_max_threshold'] = FISH_area_thresh[1]

        # µm pixel resolution
        input_params['x_pixel'] = 0.0572
        input_params['z_pixel'] = 0.2

        # distance threshold for stitching across multiple z-stacks
        input_params['distance_threshold'] = 0.75

        # Volumetric threshold for accepting a called foci, in units of µm^3
        input_params['volume_threshold'] = 0.05

        # Half the length of the cube of image data that is stored, centered on the centroid of the FISH Foci.
        # In units of xy pixels. The size of the z plane is calculated as (size_box) * (x_pixel)/(z_pixel)
        input_params['size_box'] = 25

        if not auto_call_foci:
            input_params['csv_folder'] = foci_path  # if you want to use already-called foci
            input_params['auto_call_foci_flag'] = False
        else:
            input_params['auto_call_foci_flag'] = True

        # Rename output dir name with parameters
        input_params['output_dir_name'] = input_params['output_dir_name'] +\
                                          '_size_box_' + str(input_params['size_box']) +\
                                          '_thresh_multi_' + str(input_params['threshold_multiplier'])

        # generate output
        helper.generate_output(input_params)

