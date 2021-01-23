#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 10:29:00 2021

@author: Feliks Kiszkurno
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import slopestabilitytools
import slopestabilitytools.datamanagement
import slopestabilityML
import slostabcreatedata

import pygimli as pg
import pygimli.meshtools as mt
import pygimli.physics.ert as ert

# Config
create_new_data = False
create_new_data_only = False

# Prepare folder structure for output
is_success = slopestabilitytools.folder_structure.create_folder_structure()

if not create_new_data:

    test_results = slopestabilitytools.datamanagement.import_tests()

else:

    # TODO Put this part into a function

    # Settings
    number_of_tests = 5
    rho_spread_factor = 1.5
    rho_max = 10
    layers_min = 1
    layers_max = 1
    min_depth = 4
    max_depth = 8

    # Generate parameters for tests
    # tests_horizontal = slopestabilitytools.model_params(number_of_tests,
    #                                                     rho_spread_factor, rho_max,
    #                                                     layers_min, layers_max,
    #                                                     min_depth, max_depth)

    tests_horizontal = {'hor_1': {'layer_n': 1, 'rho_values': [[1, 5], [2, 15]], 'layers_pos': np.array([-5])},
                        'hor_2': {'layer_n': 1, 'rho_values': [[1, 5], [2, 50]], 'layers_pos': np.array([-5])},
                        'hor_3': {'layer_n': 1, 'rho_values': [[1, 15], [2, 20]], 'layers_pos': np.array([-8])},
                        'hor_4': {'layer_n': 1, 'rho_values': [[1, 5], [2, 10]], 'layers_pos': np.array([-3])},
                        'hor_5': {'layer_n': 1, 'rho_values': [[1, 5], [2, 25]], 'layers_pos': np.array([-3])}}

    #  Create models and invert them
    test_results = {}

    for test_name in tests_horizontal.keys():
        test_result_curr = slostabcreatedata.create_data(test_name, tests_horizontal[test_name], max_depth)
        test_results.update({test_name: test_result_curr})
        del test_result_curr

        # Plot and save figures
        slopestabilitytools.plot_and_save(test_name, test_results[test_name], 'Test: ' + test_name)

if not create_new_data_only:
    #for test_name in test_results.keys():
        #slopestabilitytools.plot_and_save(test_name, test_results[test_name], 'Test: ' + test_name)

    ml_results = slopestabilityML.run_all_tests(test_results)
    # svm_accuracy_score, svm_accuracy_labels = slopestabilityML.svm_run(test_results)
elif create_new_data_only:
    print('Done')
