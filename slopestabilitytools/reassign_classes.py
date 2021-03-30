#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30.03.2021

@author: Feliks Kiszkurno
"""

import slopestabilitytools
import settings


def reassign_classes(test_results, class_type): # It will use number of classes defined in settings

    for test_name in test_results.keys():

        test_data = test_results[test_name]

        if class_type == 'norm':

            data_array = test_data['RESN']
            class_array = slopestabilitytools.assign_classes(data_array)
            test_data['CLASSN'] = class_array

            test_data.to_csv(settings.settings['data_folder'] + '/' + test_name + '.csv')

            test_results[test_name] = test_data

        elif class_type == 'raw':

            # Actually it is not needed right now
            print('Why? It is in CLASS column...')

        else:
            print('Incorrect type of classes! Classes has not been reassigned')
            exit(0)

    return test_results
