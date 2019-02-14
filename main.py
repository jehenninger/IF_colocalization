# #!/lab/solexa_young/scratch/jon_henninger/tools/venv/bin/python

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['font.sans-serif'] = 'Arial'
matplotlib.rcParams['font.family'] = 'sans-serif'

import methods
import grapher

# @TODO: Make heatmap of colocalization for easy representative figure generation
# @TODO: Make image and graph outputs

# NOTE: If python/matplotlib cannot find the correct font, then run the following in the python console:
# matplotlib.font_manager._rebuild()

import numpy as np
import pandas as pd
import os
import sys
from types import SimpleNamespace
import argparse
import json
from datetime import datetime
import math


# This is written so that all replicates for a given experiment are in a folder together (both .TIF and .nd files)


# parse input
parser = argparse.ArgumentParser()

input_params = methods.parse_arguments(parser)

if not os.path.isdir(input_params.parent_dir):
    print('Error: Could not read or find parent directory')
    sys.exit(0)

print('Started at: ', datetime.now())
print()

# make output directories
output_dirs = methods.make_output_directories(input_params)

# get number of experiments/sub-directories to analyze
dir_list = os.listdir(input_params.parent_dir)
dir_list.sort(reverse=False)
file_ext = ".nd"

replicate_writer = pd.ExcelWriter(os.path.join(output_dirs['parent'], 'coloc_output.xlsx'),
                                  engine='xlsxwriter')

for folder in dir_list:  # folder is a separate experiment
    if not folder.startswith('.') and \
            os.path.isdir(os.path.join(input_params.parent_dir, folder)):  # to not include hidden files or folders

            file_list = os.listdir(os.path.join(input_params.parent_dir, folder))
            base_name_files = [f for f in file_list if file_ext in f
                               and os.path.isfile(os.path.join(input_params.parent_dir, folder,  f))]
            base_name_files.sort(reverse=False)

            individual_replicate_output = pd.DataFrame(columns=['sample', 'replicate', 'channel_a', 'channel_b',
                                                                'pearson_rho', 'pearson_p-val',
                                                                'spearman_rho', 'spearman_p-val',
                                                                'manders_1', 'manders_2'])

            input_params.replicate_count_idx = 1
            for file in base_name_files:  # file is the nd file associated with a group of images for a replicate
                sample_name = file.replace(file_ext, '')
                replicate_files = [r for r in file_list if sample_name in r
                                   and os.path.isfile(os.path.join(input_params.parent_dir, folder, r))]

                replicate_files.sort(reverse=False)
                data = methods.load_images(replicate_files, input_params, folder)
                data.output_directories = output_dirs

                if data.multi_flag:
                    individual_replicate_output, data = methods.analyze_replicate(data, input_params, individual_replicate_output,
                                                                                  channel_a_idx=0, channel_b_idx=1)
                    individual_replicate_output, data = methods.analyze_replicate(data, input_params,
                                                                                  individual_replicate_output,
                                                                                  channel_a_idx=1, channel_b_idx=2)
                    individual_replicate_output, data = methods.analyze_replicate(data, input_params,
                                                                                  individual_replicate_output,
                                                                                  channel_a_idx=0, channel_b_idx=2)
                    input_params.replicate_count_idx += 1

                else:
                    individual_replicate_output, data = methods.analyze_replicate(data, input_params, individual_replicate_output)
                    input_params.replicate_count_idx += 1

            if len(individual_replicate_output) > 0:
                individual_replicate_output.to_excel(replicate_writer, sheet_name=folder[0:15], index=False)

try:
    replicate_writer = methods.adjust_excel_column_width(replicate_writer, individual_replicate_output)
    replicate_writer.save()

    methods.write_output_params(input_params)
except:
    print('Either no output or check directory structure because no experiment folders were found')

print("Finished at: ", datetime.now())
print()
print("------------------------ Completed -----------------------")
print()
