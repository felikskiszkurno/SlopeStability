#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14.04.2021

@author: Feliks Kiszkurno
"""

import pandas as pd
import numpy as np

def read_test_parameters(path_to_file):

    params = pd.read_csv(path_to_file, header=1)

    test_definitions = {}
    test_names = params['NAME'].unique()

    for name in test_names:
        params_temp = params[params['NAME'] == name]
        rho_list = []
        position = []
        layer_id = 1
        for i, row in params_temp.iterrows():
            rho_list.append([layer_id, row['RHO']])
            layer_id += 1
            if row['POS'] is not 0:
                position.append(row['POS'])
            print(row['RHO'])
        test_definitions[name] = {'layer_n': row['LAYER_N'], 'rho_values': rho_list, 'layers_pos': np.array(position)}

    return test_definitions
