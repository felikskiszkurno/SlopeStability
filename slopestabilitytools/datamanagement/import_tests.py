#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 17.01.2021

@author: Feliks Kiszkurno
"""

import slopestabilitytools.datamanagement.test_list
import settings

import pandas as pd


def import_tests(abs_path=''):
    test_results = {}
    test_names = slopestabilitytools.datamanagement.test_list('.csv', abs_path=abs_path)

    for test_name in test_names:
        test_result_curr = pd.read_csv(abs_path + settings.settings['data_folder'] + '/' + test_name + '.csv', index_col=0)
        test_results.update({test_name: test_result_curr})

    return test_results
