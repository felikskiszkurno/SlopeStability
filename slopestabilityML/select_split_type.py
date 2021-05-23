#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23.05.2021

@author: Feliks Kiszkurno
"""

import settings
import slopestabilityML


def select_split_type(test_results, random_seed):

    # Split the data set
    if settings.settings['data_split'] is 'random':

        test_training, test_prediction = slopestabilityML.split_dataset(test_results.keys(), random_seed)
        test_results_mixed = test_results

    elif settings.settings['data_split'] is 'predefined':

        test_training = test_results['training'].keys()
        test_prediction = test_results['prediction'].keys()
        test_results_mixed = {}
        test_results_mixed.update(test_results['prediction'])
        test_results_mixed.update(test_results['training'])

    return test_results_mixed, test_training, test_prediction
