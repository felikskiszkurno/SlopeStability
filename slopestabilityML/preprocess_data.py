#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19.01.2021

@author: Feliks Kiszkurno
"""

import settings
import pandas as pd
import numpy as np


def preprocess_data(data_set, *, return_x=False):

    data_set_temp = data_set.copy()

    if settings.settings['norm'] is True:
        x_train = data_set_temp.drop(['NAME', 'X', 'Z', 'INM', 'INMN', 'RES', 'CLASS', 'CLASSN', 'LABELS'], axis='columns')
    else:

        x_train = data_set_temp.drop(['NAME', 'X', 'Z', 'INM', 'INMN', 'RESN', 'CLASS', 'CLASSN', 'LABELS'], axis='columns')

    if settings.settings['sen'] is False:
        x_train = x_train.drop(['SEN'], axis='columns')

    if settings.settings['norm_class'] is True and settings.settings['use_labels'] is False:
        y_train = pd.DataFrame(data_set['CLASSN'])
    elif settings.settings['use_labels'] is True:
        y_train = pd.DataFrame(data_set['LABELS'])
    else:
        y_train = pd.DataFrame(data_set['CLASS'].to_numpy().astype(int))

    if settings.settings['depth'] is False:
        x_train = x_train.drop(['Y'], axis='columns')

    if return_x is True:
        x_positions = pd.DataFrame(data_set['X'])
        return x_train, y_train, x_positions
    elif return_x is False:
        return x_train, y_train
