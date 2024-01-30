#!/usr/bin/env python
"""
==========================
1_NormalizeSegment.py
==========================

The 1_NormalizeSegment.py integrates several interfaces to perform ...
"""

# Import necessary modules
import warnings
import sys
import time
import os
import pandas as pd
import SimpleITK as sitk
import numpy as np
import yaml

# Suppress specific warnings related to future deprecations
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Append image processing module path for imports
sys.path.append(os.path.abspath('..' + os.path.sep + 'image_processing'))

# Import custom modules for image processing
import ImageProcessing as ip
import Segmentation as seg
import FindFiles as ff


yaml_path = input("Please provide the path (incl. file name) of your yaml file: ")

# Load configuration settings from YAML file
with open(yaml_path, 'r') as file:
    config = yaml.safe_load(file)

# Record start time for script execution
start_time = time.time()

print("Script name: ", sys.argv[0])
print("Number of arguments: ", len(sys.argv))
print("The arguments are: ", str(sys.argv))

# Extract paths and configurations from loaded YAML file
path = config['inputInfo']['path']
conf_suf = config['inputImages']['conf_suf']
bg_suf = config['inputImages']['bg_suf']
ini_suf = config['inputImages']['ini_suf']
img_sufs = np.array(config['inputImages']['img_sufs'])
df_suv_files = np.array(config['inputImages']['df_suv_files'])
img_labels = np.array(config['inputImages']['img_labels'])
norm_img_sufs = np.array(config['inputImages']['norm_img_sufs'])
bg_threshs = np.array(config['processingInfo']['bg_threshs'])
max_treshs = np.array(config['processingInfo']['max_treshs'])

# Find directories of patient files
PatientDir = ff.FindDirOfFiles(np.append(np.array([conf_suf, bg_suf]), img_sufs), path)
mask_type = sitk.sitkUInt8

# Process activity weight data if available
dfs_act_weight = []
if df_suv_files.size > 0:
    for df_suv_file in df_suv_files:
        df_tmp = pd.read_excel(path + os.path.sep + df_suv_file, index_col=0)
        dfs_act_weight.append(df_tmp)

# Process each patient directory
for pat_dir in PatientDir:
    pat_name = os.path.split(os.path.dirname(pat_dir))[1]
    print(f"Processing patient: {pat_name}")
    par_results_dir = pat_dir + os.path.sep + 'results'

    # Create results directory if it doesn't exist
    if not os.path.isdir(par_results_dir):
        os.mkdir(par_results_dir)

    # Load confining, seed, and background images
    cnf_img = ip.find_load_cast(conf_suf, pat_dir, mask_type)
    seed_img = ip.find_load(ini_suf, pat_dir)
    bg_img = ip.find_load_cast(bg_suf, pat_dir, mask_type)

    # Process each image suffix and label
    for index, (img_suf, img_label) in enumerate(zip(img_sufs, img_labels)):
        img = ip.find_load(img_suf, pat_dir)

        # Skip processing if any image is missing
        if not [x for x in (img, bg_img, cnf_img) if x is None]:
            # Normalize image using background volume
            normalized_img, bg_val, sd_val = ip.norm(img, bg_img)
            ip.save_img(normalized_img, par_results_dir, 'norm' + img_label, pat_name)

            # Process SUV-normalized images if weight data is available
            if dfs_act_weight:
                norm_val = dfs_act_weight[index]['activities'][pat_name] / dfs_act_weight[index]['weights'][pat_name]
                ip.save_img(img/norm_val, par_results_dir, img_label + '_SUV', pat_name)

            # Process normalized images
            if norm_img_sufs.size > 0:
                norm_img = ip.find_load(norm_img_sufs[index], pat_dir)
                normalized_norm_img, norm_bg_val, norm_sd_val = ip.norm(norm_img, bg_img)
                rel_img = ip.remove_nan_negative_gt20(normalized_img / normalized_norm_img)
                ip.save_img(rel_img, par_results_dir, 'rel' + img_label, pat_name)
                normalized_img = rel_img

            # Initialize segmentation object
            seg_obj = seg.Segmentation(normalized_img, cnf_img, None, 1.0, seed_img, img_label)

            # Perform segmentation based on background thresholds
            if bg_threshs.size > 0:
                for bg_thresh in bg_threshs:
                    ip.save_img(seg_obj.segment_connected(bg_thresh, 1000000), par_results_dir,
                                img_label + '_seg_' + str(bg_thresh).replace('.',''), pat_name)

            # Perform segmentation based on maximum thresholds
            if max_treshs.size > 0:
                for max_tresh in max_treshs:
                    stats = sitk.LabelStatisticsImageFilter()
                    stats.Execute(normalized_img, cnf_img)
                    ip.save_img(seg_obj.segment_connected(max_tresh * stats.GetMaximum(1), 1000000), par_results_dir,
                                img_label + '_seg_' + str(max_tresh).replace('.','') + 'max', pat_name)

# Calculate and display elapsed time for script execution
elapsed_time = time.time() - start_time
print(time.strftime("Elapsed time: %H:%M:%S", time.gmtime(elapsed_time)))