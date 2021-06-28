#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19.01.2021

@author: Feliks Kiszkurno
"""

from sklearn.linear_model import SGDClassifier

import numpy as np

import slopestabilityML.plot_results
import slopestabilityML.split_dataset
import slopestabilityML.run_classification
import settings


def sgd_run(test_results, random_seed):
    # Split the data set
    test_results, test_training, test_prediction = slopestabilityML.select_split_type(test_results, random_seed)

    if settings.settings['optimize_ml'] is True:

        hyperparameters = {'loss': ['modified_huber', 'hinge', 'squared_hinge'],
                           'penalty': ['l1', 'l2'],
                           'alpha': [0.0001]}

        clf_base = SGDClassifier()

        clf = slopestabilityML.select_search_type(clf_base, hyperparameters)

    else:

        # Create classifier
        clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5, n_jobs=-1)

    # Train classifier
    results, result_class = \
        slopestabilityML.run_classification(test_training, test_prediction, test_results, clf, 'SGD')

    # Plot
    # slopestabilityML.plot_results(accuracy_labels, accuracy_score, 'SGD_prediction')
    # slopestabilityML.plot_results(accuracy_labels_training, accuracy_score_training, 'SGD_training')

    return results, result_class
