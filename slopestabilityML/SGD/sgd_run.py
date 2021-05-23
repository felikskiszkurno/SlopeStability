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

        hyperparameters = {'loss': ['modified_huber', 'perceptrone', 'hinge', 'squared_hinge'],
                           'penalty': ['l1', 'l2', 'elastinet'],
                           'alpha': [0.0001]}

        clf_base = SGDClassifier()

        clf = slopestabilityML.select_search_type(clf_base, hyperparameters)

    else:

        # Create classifier
        clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5, n_jobs=-1)

    # Train classifier
    result_class, accuracy_labels, accuracy_score, accuracy_labels_training, accuracy_score_training, depth_estim, depth_estim_accuracy, depth_estim_labels, depth_estim_training, depth_estim_accuracy_training, depth_estim_labels_training = \
        slopestabilityML.run_classification(test_training, test_prediction, test_results, clf, 'SGD')

    # Plot
    # slopestabilityML.plot_results(accuracy_labels, accuracy_score, 'SGD_prediction')
    # slopestabilityML.plot_results(accuracy_labels_training, accuracy_score_training, 'SGD_training')

    return result_class, accuracy_score, accuracy_labels, accuracy_score_training, accuracy_labels_training, depth_estim, depth_estim_accuracy, depth_estim_labels, depth_estim_training, depth_estim_accuracy_training, depth_estim_labels_training
