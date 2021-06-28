#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22.06.2021

@author: Feliks Kiszkurno
"""

from sklearn.neural_network import MLPClassifier

import numpy as np

import slopestabilityML.plot_results
import slopestabilityML.split_dataset
import slopestabilityML.run_classification
import settings


def dnn_run(test_results, random_seed):

    # Split the data set
    test_results, test_training, test_prediction = slopestabilityML.select_split_type(test_results, random_seed)

    accuracy_score = []
    accuracy_labels = []

    if settings.settings['optimize_ml'] is True:

        hyperparameters = {'activation': ['identity', 'logistic', 'tanh', 'relu'],
                           'solver': ['lbfgs', 'sgd', 'adam'],
                           }

        clf_base = MLPClassifier()

        clf = slopestabilityML.select_search_type(clf_base, hyperparameters)

    else:
        clf = MLPClassifier(solver='lbfgs')  # lbfgs is better for smaller datasets

    # Train classifier
    results, result_class = \
        slopestabilityML.run_classification(test_training, test_prediction, test_results, clf, 'DNN')

    return results, result_class
