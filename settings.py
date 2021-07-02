#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 26.03.2021

@author: Feliks Kiszkurno
"""

import os
import slopestabilitytools
from datetime import datetime


def init():

    global settings

    settings = {}

    # Paths
    settings['base_folder'] = os.getcwd()
    settings['results_folder'] = os.path.join(settings['base_folder'], 'results')
    settings['data_folder'] = os.path.join(settings['results_folder'], 'data')
    settings['data_folder_grd'] = os.path.join(settings['results_folder'], 'data_grd')
    settings['data_measurement'] = os.path.join(settings['results_folder'], 'data_measurement')
    settings['figures_folder'] = os.path.join(settings['results_folder'], 'figures')
    settings['clf_folder'] = os.path.join(settings['results_folder'], 'classifiers')

    # Log file
    settings['log_file_name'] = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

    # Training and prediction split
    settings['split_proportion'] = 0.5  # Part of available profiles that will be used for prediction
    settings['data_split'] = 'predefined' # 'random' or 'predefined'
    settings['use_batches'] = True  # True or False
    if settings['use_batches'] is True:
        settings['data_split'] = 'predefined'

    settings['retrain_clf'] = False  # True trains classifiers for each batch separately
    settings['reuse_clf'] = True  # Load classifiers if they exist in classifiers folder
    if settings['reuse_clf'] is True:
        settings['clf_trained'] = slopestabilitytools.find_clf()  # List of trained classifiers, they won't be retrained unless retrain_clf is set to True
    else:
        settings['clf_trained'] = []
    # Interpolate results to grid inside create_data script
    settings['grd'] = True

    # Sample weight
    settings['weight'] = False

    # Parameters for resampling
    settings['resample'] = False
    settings['resample_x_spacing'] = 1
    settings['resample_y_spacing'] = 1

    # Reduce sample population
    settings['reduce_samples'] = False
    settings['reduce_samples_factor'] = 0.25

    # Normalization and classes
    settings['norm_class'] = True  # True to use normalized classes, False to use class_ids
    settings['norm_class_num'] = 2  # Number of classes for normalized data
    settings['norm'] = True  # True to use normalized data, False to use raw data
    settings['use_labels'] = False  # True to use labels instead of classes

    # Ignore data points with insufficient sensitivity
    settings['min_sen_pred'] = True
    settings['min_sen_pred_val'] = 0.3
    settings['min_sen_train'] = True
    settings['min_sen_train_val'] = 0.3

    # Include sensitivity
    settings['sen'] = True  # True - include sensitivity, False - ignore sensitivity

    # Include depth
    settings['depth'] = True   # True - include depth, False - ignore depth

    # Borehole simulation
    settings['sim_bh'] = True
    settings['bh_pos'] = {1: {'x_start': -17, 'x_end': -15, 'y_start': -18, 'y_end': 0},
                          2: {'x_start': 0, 'x_end': 2, 'y_start': -18, 'y_end': 0}}

    # Balance classes
    settings['balance'] = False

    # Classifiers
    settings['optimize_ml'] = True  # True - performs hyperparameter search
    settings['optimize_ml_type'] = 'exhaustive'  # Type of grid search exhaustive or halved

    # Plots
    settings['plot_formats'] = ['png']  # list of formats to save plots as, supported formats: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff


    ## LEGACY STUFF - SET TO FALSE UNLESS THERE IS A VERY GOOD REASON!!!!
    # Clip data to max and min values from the input model
    settings['clip'] = False  # True - clip data, False - use unclipped data