#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 26.03.2021

@author: Feliks Kiszkurno
"""


def init():

    global settings

    settings = {}

    # Normalization and classes
    settings['norm_class'] = True  # True to use normalized classes, False to use class_ids
    settings['norm_class_num'] = 5  # Number of classes for normalized data
    settings['norm'] = False  # True to use normalized data, False to use raw data

    # Include sensitivity
    settings['sen'] = False  # True - include sensitivity, False - ignore sensitivity

    # Paths
    settings['results_folder'] = "results"

    # Plots
    settings['plot_formats'] = ['png']
