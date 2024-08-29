#!/usr/bin/env python
"""
==========================
RadiomicsFct.py
==========================

This module integrates several interfaces to perform radiomics feature extraction on medical images.
"""

# Import necessary libraries
from radiomics import featureextractor, getTestCase
import SimpleITK as sitk
import six
import sys
import os
import numpy as np
import pandas as pd

# Ensure the image processing modules are in the path
sys.path.append(os.path.abspath('..' + os.path.sep + 'image_processing'))
import ImageProcessing as ip
import FindFiles as ff
import yaml


def do_radiomics(pat_dirs, img_suf, mask_suf, bin_width, out_path, par_name,
                 shape=True, check_ini=None, yaml_file='./yaml-files/RadiomicsFct_VOI.yaml',
                 do_interp=True):
    """
    Performs radiomics analysis on a set of patient directories.
    """
    # Get the location of the example settings file
    params_file_voi = os.path.abspath(yaml_file)
    with open(params_file_voi, 'r') as file:
        config_voi = yaml.safe_load(file)

    # Initialize variables
    voi_label = 1  # Label for the volume of interest
    first_col0 = 22  # Starting index for columns in the DataFrame
    if do_interp:
        first_col0 += 15  # Adjust column index if interpolation is enabled

    df_list = []  # List to hold data for DataFrame
    pat_names = []  # List to store patient names
    first_data = True  # Flag to indicate the first iteration
    df_col_group = []
    df_col_feature = []

    # Iterate over each patient directory
    for pat_dir in pat_dirs:
        # Load the mask image and extract patient name
        mask_image = ip.find_load(mask_suf, pat_dir)
        if mask_image is None:
            mask_image = ip.find_load(mask_suf, os.path.dirname(pat_dir))
        pat_name = mask_image.GetMetaData("0010|0010")
        print(pat_name)  # Output the patient name

        # Check for initialization file if specified
        check_ini_tmp = check_ini
        if check_ini_tmp is not None:
            ini_path = ff.FindPat(check_ini_tmp, pat_dir)
            if ini_path != []:
                check_ini_tmp = None

        # Load the input image
        inputImage = ip.find_load(img_suf, pat_dir)
        if inputImage is None:
            continue

        inputImage = ip.mask_vals(inputImage, below=0, above=bin_width*200)
        pat_names.append(pat_name)

        # Initialize variables for VOI processing
        voi_size = 0
        if mask_image is not None and check_ini_tmp is None:
            # Get x dimension
            voxel_dim_x = mask_image.GetSpacing()[0]

            # Calculate VOI size
            stats = sitk.StatisticsImageFilter()
            stats.Execute(mask_image)
            voi_size = int(stats.GetSum())

            if voi_size > 18:
                # https://pyradiomics.readthedocs.io/en/latest/customization.html
                # Initialize the radiomics feature extractor with settings
                extractor = featureextractor.RadiomicsFeatureExtractor(params_file_voi)
                if do_interp:
                    if 'resampledPixelSpacing' in config_voi['setting']:
                        # Set interpolation settings to in-plane resolution
                        extractor.settings['resampledPixelSpacing'] = np.array(config_voi['setting']['resampledPixelSpacing'])
                    else:
                        # Set interpolation settings to in-plane resolution
                        extractor.settings['resampledPixelSpacing'] = [voxel_dim_x, voxel_dim_x, voxel_dim_x]
                else:
                    extractor.settings['resampledPixelSpacing'] = None
                extractor.settings['binWidth'] = bin_width
                extractor.enableFeatureClassByName('shape', enabled=shape)

                # Execute feature extraction
                result = extractor.execute(inputImage, mask_image, label=voi_label)

                # Process and store the results
                if first_data:
                    df_col_group = [key[key.find('_') + 1:key.find('_', key.find('_') + 1)] for key in result.keys()][
                                   first_col0:]
                    df_col_feature = [key[key.find('_', key.find('_') + 1) + 1:] for key in result.keys()][first_col0:]

                df_list.append([val for val in result.values()][first_col0:])
                first_data = False

            else:
                df_list.append([pat_name, 'VOI size lower or equal to 18'])
        else:
            df_list.append([pat_name, 'No ini file'])

    # Save the results to an Excel file
    save_results_to_excel(df_list, df_col_group, df_col_feature, pat_names, out_path, par_name, voi_label)


def save_results_to_excel(df_list, df_col_group, df_col_feature, pat_names, out_path,
                          par_name, voi_label):
    """
    Saves the extracted radiomics features to an Excel file.
    """
    writer = pd.ExcelWriter(os.path.join(out_path, f'{par_name}_Radiomics.xlsx'), engine='xlsxwriter')
    col0 = [par_name] * len(df_col_group)
    # tuples = list(zip(*[col0, [''] * len(df_col_group), df_col_feature]))
    tuples = list(zip(*[col0, df_col_group, df_col_feature]))
    multi_ind_columns = pd.MultiIndex.from_tuples(tuples, names=('Parameter', 'Group', 'Feature'))

    df = pd.DataFrame(df_list, columns=multi_ind_columns, index=pat_names)

    df_empty = pd.DataFrame([], columns=multi_ind_columns)
    df_empty.to_excel(writer, f"VOI_{voi_label}")

    def convert_to_float(x):
        if x is None:
            return x
        try:
            return float(x)
        except ValueError:
            return x

    df = df.applymap(convert_to_float)
    df.to_excel(writer, f"VOI_{voi_label}", header=False, startrow=2)
    writer.close()


def do_voxel_radiomics(pat_dirs, img_suf, mask_suf1=None, mask_suf2=None, bin_width=0.1,
                       par_name='', dilate_rad=0, yaml_file='./yaml-files/RadiomicsFct_Voxel.yaml',
                       do_interp=True):
    """
        Performs voxel-wise radiomics analysis on a set of patient directories.
    """
    # https://pyradiomics.readthedocs.io/en/latest/customization.html
    # Get the location of the example settings file
    params_file_voxel = os.path.abspath('RadiomicsFct_Voxel.yaml')
    with open(params_file_voxel, 'r') as file:
        config_voxel = yaml.safe_load(file)

    # Initialize a SimpleITK filter to cast images to uint8
    cast_fil_i8 = sitk.CastImageFilter()
    cast_fil_i8.SetOutputPixelType(sitk.sitkUInt8)

    # todo: application of different label numbers
    label_n = 1  # Define the label number for the mask
    pat_names = []  # List to store patient names

    # Process only the first patient directory for demonstration
    for pat_dir in pat_dirs[:1]:
        # Load the input image using a custom function
        input_img = ip.find_load(img_suf, pat_dir)
        pat_name = input_img.GetMetaData("0010|0010")  # Extract patient name from metadata
        print(pat_name)  # Print the patient name
        pat_names.append(pat_name)  # Add patient name to the list

        # Determine voxel dimensions and create an output directory for texture images
        voxel_dim_x = input_img.GetSpacing()[0]
        out_path = pat_dir + os.path.sep + "texture_imgs"
        if not os.path.isdir(out_path):
            os.mkdir(out_path)

        # Generate or load mask images
        mask_img = None
        if mask_suf1 is None:
            # Generate a mask image based on the input image
            mask_img = ip.gen_mask_above_val(input_img, 0.1)
            ip.save_img(mask_img, out_path, 'Mask', pat_name)
        else:
            # Load the first mask image
            mask_img = ip.find_load(mask_suf1, pat_dir)
            if mask_suf2 is not None:
                # Load and combine the second mask image if provided
                mask_img2 = ip.find_load(mask_suf2, pat_dir)
                mask_img = ip.union_vois(mask_img, mask_img2)
            # Cast the mask image to uint8
            mask_img = cast_fil_i8.Execute(mask_img)

        # Apply the mask to the input image
        mask_img_f = sitk.MaskImageFilter()
        mask_img_f.SetGlobalDefaultCoordinateTolerance(0.001)
        masked_img = mask_img_f.Execute(input_img, mask_img)
        # Save the masked image and the mask
        ip.save_img(mask_img, out_path, 'Mask', pat_name)
        ip.save_img(masked_img, out_path, par_name + '_masked', pat_name)

        # Dilate the mask if a dilation radius is specified
        if dilate_rad != 0:
            mask_img = sitk.BinaryDilate(mask_img, (dilate_rad, dilate_rad, dilate_rad))

        # Initialize the radiomics feature extractor with the YAML configuration
        extractor = featureextractor.RadiomicsFeatureExtractor(params_file_voxel)
        # Set interpolation and bin width settings
        if do_interp:
            if 'resampledPixelSpacing' in config_voxel['setting']:
                # Set interpolation settings to in-plane resolution
                extractor.settings['resampledPixelSpacing'] = np.array(config_voxel['setting']['resampledPixelSpacing'])
            else:
                # Set interpolation settings to in-plane resolution
                extractor.settings['resampledPixelSpacing'] = [voxel_dim_x, voxel_dim_x, voxel_dim_x]
        else:
            extractor.settings['resampledPixelSpacing'] = None
        extractor.settings['binWidth'] = bin_width
        extractor.settings['initValue'] = np.NAN

        # Execute voxel-based feature extraction
        featureVector = extractor.execute(input_img, mask_img, label=label_n, voxelBased=True)

        # Save and print the results
        for featureName, featureValue in six.iteritems(featureVector):
            if isinstance(featureValue, sitk.Image):
                # Save each voxel-based feature image
                ip.save_img(featureValue, out_path, par_name + '_' + featureName, pat_name)
                print(f'Computed {par_name}_{featureName}, stored as "{pat_name}_{par_name}_{featureName}.nii"')
            else:
                print(f'{par_name}_{featureName}: {featureValue}')
