#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19.01.2021

@author: Feliks Kiszkurno
"""

import settings
import pandas as pd


def preprocess_data(data_set):

    if settings.settings['norm'] is True:
        x_train = data_set.drop(['X', 'Z', 'INM', 'INMN', 'RES', 'CLASS', 'CLASSN'], axis='columns')
    else:
        x_train = data_set.drop(['X', 'Z', 'INM', 'INMN', 'RESN', 'CLASS', 'CLASSN'], axis='columns')

    if settings.settings['sen'] is False:
        x_train = x_train.drop(['SEN'], axis='columns')

    if settings.settings['norm_class'] is True:
        y_train = pd.DataFrame(data_set['CLASSN'])
    else:
        y_train = pd.DataFrame(data_set['CLASS'])

    if settings.settings['depth'] is False:
        x_train = x_train.drop(['Y'], axis='columns')

    return x_train, y_train
