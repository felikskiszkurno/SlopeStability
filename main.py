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
import gc
import numpy as np
import settings
import test_definitions

from datetime import datetime

settings.init()

# Config
create_new_data = False # set to True if you need to reassign the classes
invert_existing_data = False  # invert existing measurements
create_new_data_only = False  # set to False in order to run ML classifications
reassign_classes = False; class_type = 'norm'
param_path = os.path.abspath(os.path.join(os.getcwd()) + '/' + 'TestDefinitions/hor1__final_500_50.csv')
test_definitions.init(path=param_path)

# Load existing data instead of creating new one.
print('start')
if not create_new_data:
    if settings.settings['data_split'] is 'random':
        test_results = slopestabilitytools.datamanagement.import_tests()
    elif settings.settings['data_split'] is 'predefined':
        test_results = slopestabilitytools.datamanagement.import_tests_predefined()

    else:
        print('Error: undefined tests')
        exit(0)

    # if reassign_classes is True:
    #        test_results = slopestabilitytools.reassign_classes(test_results, class_type)

    # Check if folder structure for figures exists and create it if not
    if settings.settings['use_batches'] is True:

        is_success = slopestabilitytools.folder_structure.create_folder_structure(batch_names=test_results['prediction'].keys())

    elif settings.settings['use_batches'] is False:

        is_success = slopestabilitytools.folder_structure.create_folder_structure()

# Create new data
else:

    if invert_existing_data is False:
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
        # tests_parameters = slopestabilitytools.model_params(number_of_tests,
        #                                                     rho_spread_factor, rho_max,
        #                                                     layers_min, layers_max,
        #                                                     min_depth, max_depth)

        #tests_parameters = test_definitions.test_definitions


        # tests_parameters = {'hor_11': {'layer_n': 1, 'rho_values': [[1, 10], [2, 12]], 'layers_pos': np.array([-4])}}

        #  Create models and invert them
        test_results = {}
        test_results_grd = {}
        print(test_definitions.test_parameters)
        for test_name in test_definitions.test_parameters.keys():
            test_result_curr, test_result_curr_grd, test_rho_max, test_rho_min = slostabcreatedata.create_data(test_name,
                                                                                         test_definitions.test_parameters[test_name],
                                                                                         abs(test_definitions.test_parameters[test_name]['layers_pos'].max()),
                                                                                         lambda_param=test_definitions.test_parameters[test_name]['lambda'][0],
                                                                                         z_weight=test_definitions.test_parameters[test_name]['z_weight'][0])

            test_results.update({test_name: test_result_curr})
            test_results_grd.update({test_name: test_result_curr_grd})
            del test_result_curr, test_result_curr_grd

            # Plot and save figures
            #slopestabilitytools.plot_and_save(test_name, test_results[test_name], 'Test: ' + test_name, test_rho_max,
            #                                  test_rho_min)

            slopestabilitytools.plot_and_save(test_name + '_grd', test_results_grd[test_name], 'Test: ' + test_name + '_grd', test_rho_max,
                                              test_rho_min)

            gc.collect()

    elif invert_existing_data is True:

        profile_names = slopestabilitytools.datamanagement.test_list('.ohm',
                                                                 abs_path=settings.settings['data_measurement'])

        for profile_name in profile_names:
            slostabcreatedata.invert_data(profile_name)

        # Evaluate data with ML techniques
if not create_new_data_only:

    log_file_name = settings.settings['log_file_name']
    log_file = open(os.path.join(settings.settings['results_folder'], log_file_name), 'w')
    log_file.write('Starting log file...')
    log_file.write('Started at: ' + datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
    log_file.close()

    print('Running ML stuff...')
    ml_results = slopestabilityML.run_all_tests(test_results)

    log_file = open(os.path.join(settings.settings['results_folder'], log_file_name), 'a')
    log_file.write('Finishing log file...')
    log_file.write('Finished at: ' + datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
    log_file.close()

# Finish the script if ML classifiaction was not executed
elif create_new_data_only:

    print('Done')
