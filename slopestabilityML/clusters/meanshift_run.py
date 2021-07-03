#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 03.07.2021

@author: Feliks Kiszkurno
"""

from sklearn.cluster import MeanShift

import numpy as np

import slopestabilityML.plot_results
import slopestabilityML.split_dataset
import slopestabilityML.run_classification

import settings


def meanshift_run(test_results, random_seed):
    # Split the data set
    test_results, test_training, test_prediction = slopestabilityML.select_split_type(test_results, random_seed)

    if settings.settings['optimize_ml'] is True:

        hyperparameters = {'cluster_all': [True, False],
                           'bin_seeding': [True, False]}

        clf_base = MeanShift()

        clf = slopestabilityML.select_search_type(clf_base, hyperparameters)
        clf = MeanShift(cluster_all=False, bin_seeding=True)

    else:
        # Create classifier
        clf = MeanShift(cluster_all=False, bin_seeding=True)

    # Train classifier
    results, result_class = \
        slopestabilityML.run_classification(test_training, test_prediction, test_results, clf, 'MeanShift')

    # Plot
    # slopestabilityML.plot_results(accuracy_labels, accuracy_score, 'GBC_prediction')
    # slopestabilityML.plot_results(accuracy_labels_training, accuracy_score_training, 'GBC_training')

    return results, result_class
