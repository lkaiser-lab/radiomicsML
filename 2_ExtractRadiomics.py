#!/usr/bin/env python
"""
==========================
2_ExtractRadiomics.py
==========================

The 2_ExtractRadiomics.py integrates several interfaces to perform ...
"""

# import necessary modules
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import sys
import time
import os
import RadiomicsFct as rf
import yaml

sys.path.append(os.path.abspath('..'+os.path.sep+'image_processing'))
print(os.path.abspath('..'+os.path.sep+'image_processing'))
import FindFiles as ff


yaml_path = input("Please provide the path (incl. file name) of your yaml file: ")

with open(yaml_path, 'r') as file:
    config = yaml.safe_load(file)

#main
start_time = time.time()

print("Script name: ", sys.argv[0])
print("Number of arguments: ", len(sys.argv))
print("The arguments are: ", str(sys.argv))

path = config['general_settings']['path']
out_path = config['general_settings']['out_path']

if not os.path.isdir(out_path):
    os.mkdir(out_path)

# Extracting the entries for img, vol, and bin_width
img_sufs = config.get('images', {}).get('img_sufs', {})
vol_sufs = config.get('images', {}).get('vol_sufs', {})
bin_widths = config.get('images', {}).get('bin_widths', {})

PatientDir = ff.FindDirOfFiles(list(img_sufs.values()), path)

# Looping through the img_sufs entries
for suffix_name, file_pattern in img_sufs.items():
    print(f"{suffix_name}")
    img_suf = img_sufs[suffix_name]
    vol_suf = vol_sufs[suffix_name]
    bin_width = bin_widths[suffix_name]

    rf.do_radiomics(PatientDir, img_sufs[suffix_name],
                           vol_sufs[suffix_name],
                           bin_widths[suffix_name],
                           out_path, suffix_name,
                           shape=config['general_settings']['do_shape'],
                           yaml_file=config['general_settings']['radiomics_yaml_file'],
                           do_interp=config['general_settings']['do_interpolation'])

elapsed_time = time.time() - start_time
print(time.strftime("Elapsed time: %H:%M:%S", time.gmtime(elapsed_time)))