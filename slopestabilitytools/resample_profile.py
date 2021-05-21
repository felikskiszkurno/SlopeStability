#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19.05.2021

@author: Feliks Kiszkurno
"""

import pandas as pd
import numpy as np
from scipy import interpolate

import settings
import test_definitions
import slopestabilitytools


def resample_profile(profile_data):

    x_values = np.arange(np.ceil(np.min(profile_data['X'])),
                      np.floor(np.max(profile_data['X'])),
                      settings.settings['resample_x_spacing'])

    y_values = np.arange(np.ceil(np.min(profile_data['Y'])),
                      np.ceil(np.max(profile_data['Y'])),
                      settings.settings['resample_y_spacing'])

    x_new = []
    y_new = []

    for x in x_values:
        x_new.extend(x*np.ones([y_values.size]))
        y_new.extend(y_values)

    profile_data_resampled = pd.DataFrame()

    profile_data_resampled['NAME'] = [profile_data['NAME'][0]]*len(x_new)  # Assumes there is only one name
    profile_data_resampled['X'] = x_new
    profile_data_resampled['Y'] = y_new

    if settings.settings['norm'] is True:
        labels = ['INMN', 'RESN', 'SEN']
    elif settings.settings['norm'] is False:
        labels = ['INM', 'RES', 'SEN']
    else:
        labels = ['INM', 'INMN', 'RES', 'RESN', 'SEN']

    for data_column in labels:
        interp_f = interpolate.interp2d(profile_data['X'], profile_data['Y'], profile_data[data_column])
        profile_data_resampled[data_column] = interp_f(x_values, y_values).reshape([len(x_values)*len(y_values)])


    resistivity_map = test_definitions.test_parameters[profile_data['NAME'][0]]['rho_values']

    if settings.settings['norm'] is True:
        input_model = profile_data_resampled['INMN']
    elif settings.settings['norm'] is False:
        input_model = profile_data_resampled['INM']

    classes = slopestabilitytools.assign_class01(input_model, resistivity_map)
    profile_data_resampled['CLASS'] = classes

    classesn = slopestabilitytools.assign_classes(slopestabilitytools.normalize(input_model))
    profile_data_resampled['CLASSN'] = classesn

    return profile_data_resampled
