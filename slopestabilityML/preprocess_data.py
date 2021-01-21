#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19.01.2021

@author: Feliks Kiszkurno
"""

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def preprocess_data(data_set):

    x_train = data_set.drop(['Z', 'INM', 'CLASS'], axis='columns')
    y_train = data_set['CLASS']

    num_feat = []

    return x_train, y_train
