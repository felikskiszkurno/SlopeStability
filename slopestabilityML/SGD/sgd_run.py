#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19.01.2021

@author: Feliks Kiszkurno
"""

from sklearn.linear_model import SGDClassifier

import slopestabilityML.plot_results
import slopestabilityML.split_dataset
import slopestabilityML.run_classification
import settings


def sgd_run(test_results, random_seed):

    # Split the data set
    if settings['data_split'] is 'random':
        test_training, test_prediction = slopestabilityML.split_dataset(test_results.keys(), random_seed)
        test_results_mixed = test_results
    elif settings['data_split'] is 'predefined':
        test_training = test_results['training'].keys()
        test_prediction = test_results['prediction'].keys()
        test_results_mixed = {}
        test_results_mixed.update(test_results['prediction'])
        test_results_mixed.update(test_results['training'])

    # Create classifier
    clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5, n_jobs=-1)

    # Train classifier
    result_class, accuracy_labels, accuracy_score, accuracy_labels_training, accuracy_score_training, depth_estim, depth_estim_accuracy, depth_estim_labels, depth_estim_training, depth_estim_accuracy_training, depth_estim_labels_training = \
        slopestabilityML.run_classification(test_training, test_prediction, test_results_mixed, clf, 'SGD')

    # Plot
    # slopestabilityML.plot_results(accuracy_labels, accuracy_score, 'SGD_prediction')
    # slopestabilityML.plot_results(accuracy_labels_training, accuracy_score_training, 'SGD_training')

    return result_class, accuracy_score, accuracy_labels, accuracy_score_training, accuracy_labels_training, depth_estim, depth_estim_accuracy, depth_estim_labels, depth_estim_training, depth_estim_accuracy_training, depth_estim_labels_training
