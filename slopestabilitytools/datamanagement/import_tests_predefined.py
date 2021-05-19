#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19.01.2021

@author: Feliks Kiszkurno
"""

import slopestabilitytools.datamanagement.test_list
import settings

import pandas as pd


def import_tests_predefined(abs_path=''):

    test_results_training = {}
    test_names_training = slopestabilitytools.datamanagement.test_list('.csv', abs_path=abs_path + settings.settings[
        'data_folder'] + 'training/')

    for test_name in test_names_training:
        test_result_curr = pd.read_csv(abs_path + settings.settings['data_folder'] + 'training/' + test_name + '.csv', index_col=0)
        test_results_training.update({test_name: test_result_curr})

    del test_name, test_result_curr

    test_results_prediction = {}
    test_names_prediction = slopestabilitytools.datamanagement.test_list('.csv', abs_path=abs_path + settings.settings[
        'data_folder'] + 'prediction/')

    for test_name in test_names_prediction:
        test_result_curr = pd.read_csv(abs_path + settings.settings['data_folder'] + 'prediction/' + test_name + '.csv', index_col=0)
        test_results_prediction.update({test_name: test_result_curr})

    test_results = {'training': test_results_training,
                    'prediction': test_results_prediction}

    return test_results
