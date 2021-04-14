#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 26.03.2021

@author: Feliks Kiszkurno
"""


def init():

    global settings

    settings = {}

    settings['split_proportion'] = 0.25

    # Normalization and classes
    settings['norm_class'] = True  # True to use normalized classes, False to use class_ids
    settings['norm_class_num'] = 5  # Number of classes for normalized data
    settings['norm'] = False  # True to use normalized data, False to use raw data
    settings['use_labels'] = False  # True to use labels instead of classes

    # Include sensitivity
    settings['sen'] = True  # True - include sensitivity, False - ignore sensitivity

    # Include depth
    settings['depth'] = True   # True - include depth, False - ignore depth

    # Paths
    settings['results_folder'] = 'results'
    settings['data_folder'] = settings['results_folder'] + '/data/'
    settings['figures_folder'] = settings['results_folder'] + '/figures/'

    # Plots
    settings['plot_formats'] = ['png']  # list of formats to save plots as, supported formats: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff


    ## LEGACY STUFF - SET TO FALSE UNLESS THERE IS A VERY GOOD REASON!!!!
    # Clip data to max and min values from the input model
    settings['clip'] = False  # True - clip data, False - use unclipped data
