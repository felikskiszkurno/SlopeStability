#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 21.06.2021

@author: Feliks Kiszkurno
"""

from sklearn.utils import resample
import pandas as pd
import numpy as np


def balance_classes(test_results, *, test_name='name'):

    col_names = test_results.columns.values.tolist()

    test_results_orig = test_results.copy()

    test_results = test_results.drop(columns=['NAME'])

    classes_list = np.unique(test_results['CLASSN'].to_numpy())
    classes_count = np.zeros(classes_list.shape)

    if len(classes_list) != 1:
        for class_id in range(len(classes_list)):
            classes_count[class_id] = len(test_results[test_results['CLASSN'] == classes_list[class_id]].index)

        classes_count_max = np.max(classes_count)
        classes_max = classes_list[np.where(classes_count == classes_count_max)][0]

        test_results_resamp = pd.DataFrame(columns=test_results.columns.values.tolist())

        for class_id in range(len(classes_list)):

            test_results_temp = test_results[test_results['CLASSN'] == class_id].copy()
            test_results_temp = test_results_temp.drop(columns=['CLASSN'])

            test_results_temp_resamp = resample(test_results_temp,
                                                replace=True,
                                                n_samples=int(classes_count_max))

            test_results_temp_resamp['CLASSN'] = np.ones(len(test_results_temp_resamp.index))*classes_list[class_id]

            test_results_resamp = pd.concat([test_results_resamp, test_results_temp_resamp])

        test_results_resamp['NAME'] = [test_name]*len(test_results_resamp.index)

        test_results_resamp = test_results_resamp[col_names]

    else:
        test_results_resamp = test_results_orig.copy()

    return test_results_resamp

