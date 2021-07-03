#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 03.07.2021

@author: Feliks Kiszkurno
"""

from sklearn.cluster import KMeans

import numpy as np

import slopestabilityML.plot_results
import slopestabilityML.split_dataset
import slopestabilityML.run_classification

import settings


def kmeans_run(test_results, random_seed):
    # Split the data set
    test_results, test_training, test_prediction = slopestabilityML.select_split_type(test_results, random_seed)

    if settings.settings['optimize_ml'] is True:

        hyperparameters = {#'n_clusters': list(np.arange(1, 10, 1)),
                           'algorithm': ['full', 'elkan'],
                           'init': ['k-means++', 'random']}

        clf_base = KMeans(n_clusters=settings.settings['norm_class_num'])

        clf = slopestabilityML.select_search_type(clf_base, hyperparameters)

    else:
        # Create classifier
        clf = KMeans(n_clusters=settings.settings['norm_class_num'], init='k-means++', algorithm='elkan')

    # Train classifier
    results, result_class = \
        slopestabilityML.run_classification(test_training, test_prediction, test_results, clf, 'KMeans')

    # Plot
    # slopestabilityML.plot_results(accuracy_labels, accuracy_score, 'GBC_prediction')
    # slopestabilityML.plot_results(accuracy_labels_training, accuracy_score_training, 'GBC_training')

    return results, result_class
