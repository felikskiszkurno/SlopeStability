#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 10:29:00 2021

@author: Feliks Kiszkurno
"""

import slopestabilitytools.datamanagement
import slopestabilitytools
import slopestabilityML
import slostabcreatedata
import os.path
import numpy as np
import settings
import test_definitions

settings.init()
test_definitions.init()
# Config
create_new_data = True  # set to True if you need to reassign the classes
create_new_data_only = True  # set to False in order to run ML classifications
reassign_classes = False; class_type = 'norm'

# Load existing data instead of creating new one.
if not create_new_data:

    test_results = slopestabilitytools.datamanagement.import_tests()

    # if reassign_classes is True:
    #        test_results = slopestabilitytools.reassign_classes(test_results, class_type)

    # Check if folder structure for figures exists and create it if not
    is_success = slopestabilitytools.folder_structure.create_folder_structure()

# Create new data
else:

    # Prepare folder structure for output
    is_success = slopestabilitytools.folder_structure.create_folder_structure()

    # TODO Put this part into a function

    # Settings
    number_of_tests = 5
    rho_spread_factor = 1.5
    rho_max = 20
    layers_min = 1
    layers_max = 2
    min_depth = 4
    max_depth = 8

    # Generate parameters for tests
    # tests_horizontal = slopestabilitytools.model_params(number_of_tests,
    #                                                     rho_spread_factor, rho_max,
    #                                                     layers_min, layers_max,
    #                                                     min_depth, max_depth)

    #tests_horizontal = test_definitions.test_definitions
    tests_horizontal = slostabcreatedata.read_test_parameters(os.path.abspath(os.path.join(os.getcwd()) + '/' + 'TestDefinitions/hor_1layer_varying_depth.csv'))


    # tests_horizontal = {'hor_11': {'layer_n': 1, 'rho_values': [[1, 10], [2, 12]], 'layers_pos': np.array([-4])}}

    #  Create models and invert them
    test_results = {}

    for test_name in tests_horizontal.keys():
        test_result_curr, test_rho_max, test_rho_min = slostabcreatedata.create_data(test_name,
                                                                                     tests_horizontal[test_name],
                                                                                     abs(tests_horizontal[test_name]['layers_pos'].max()))
        test_results.update({test_name: test_result_curr})
        del test_result_curr

        # Plot and save figures
        slopestabilitytools.plot_and_save(test_name, test_results[test_name], 'Test: ' + test_name, test_rho_max,
                                          test_rho_min)

# Evaluate data with ML techniques
if not create_new_data_only:

    print('Running ML stuff...')
    ml_results = slopestabilityML.run_all_tests(test_results)

# Finish the script if ML classifiaction was not executed
elif create_new_data_only:

    print('Done')
