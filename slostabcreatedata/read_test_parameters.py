#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14.04.2021

@author: Feliks Kiszkurno
"""

import pandas as pd
import numpy as np

def read_test_parameters(path_to_file):

    params = pd.read_csv(path_to_file)

    test_definitions = {}
    test_names = params['NAME'].unique()

    for name in test_names:
        params_temp = params[params['NAME'] == name]
        rho_list = []
        position = []
        lambda_value = []; lambda_temp = []
        z_weight = []; z_temp = []
        layer_id = 1
        for i, row in params_temp.iterrows():
            rho_list.append([layer_id, row['RHO']])
            lambda_temp.append(row['LAMBDA'])
            z_temp.append(row['Z_WEIGHT'])
            layer_id += 1
            if int(row['POS']) is not 0:
                position.append(row['POS'])
        lambda_value.append(np.unique(row['LAMBDA'])[0])
        z_weight.append(np.unique(row['Z_WEIGHT'])[0])
        test_definitions[name] = {'layer_n': row['LAYER_N'], 'rho_values': rho_list, 'layers_pos': np.array(position),
                                  'lambda': lambda_value, 'z_weight': z_weight}

    return test_definitions
