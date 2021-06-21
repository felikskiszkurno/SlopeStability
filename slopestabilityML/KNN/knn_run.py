#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 26.01.2021

@author: Feliks Kiszkurno
"""

from sklearn.neighbors import KNeighborsClassifier

import numpy as np

import slopestabilityML.plot_results
import slopestabilityML.split_dataset
import slopestabilityML.run_classification
import settings


def knn_run(test_results, random_seed):
    # Split the data set
    test_results, test_training, test_prediction = slopestabilityML.select_split_type(test_results, random_seed)

    if settings.settings['optimize_ml'] is True:

        hyperparameters = {'neighbours': list(np.arange(2, 9, 1)),
                           'weights': ['uniform', 'distance'],
                           'p': [1, 2]}

        clf_base = KNeighborsClassifier()

        clf = slopestabilityML.select_search_type(clf_base, hyperparameters)

    else:
        # Create classifier
        clf = KNeighborsClassifier(n_neighbors=2, n_jobs=-1)

    # Train classifier
    results, result_class = \
        slopestabilityML.run_classification(test_training, test_prediction, test_results, clf, 'KNN')

    # Plot
    # slopestabilityML.plot_results(accuracy_labels, accuracy_score, 'KNN_prediction')
    # slopestabilityML.plot_results(accuracy_labels_training, accuracy_score_training, 'KNN_training')

    return results, result_class
