#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 26.03.2021

@author: Feliks Kiszkurno
"""

import slopestabilityML.plot_results
import slopestabilityML.split_dataset
import slopestabilityML.run_classification

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

import settings


def adaboost_run(test_results, random_seed):

    # Split the data set
    if settings.settings['data_split'] is 'random':
        test_training, test_prediction = slopestabilityML.split_dataset(test_results.keys(), random_seed)
        test_results_mixed = test_results
    elif settings['data_split'] is 'predefined':
        test_training = test_results['training'].keys()
        test_prediction = test_results['prediction'].keys()
        test_results_mixed = {}
        test_results_mixed.update(test_results['prediction'])
        test_results_mixed.update(test_results['training'])

    clf = AdaBoostClassifier()#base_estimator=DecisionTreeClassifier(max_depth=3),
                             #n_estimators=20, random_state=0)

    # Train classifier
    result_class, accuracy_labels, accuracy_score, accuracy_labels_training, accuracy_score_training, depth_estim, depth_estim_accuracy, depth_estim_labels, depth_estim_training, depth_estim_accuracy_training, depth_estim_labels_training = \
        slopestabilityML.run_classification(test_training, test_prediction, test_results_mixed, clf, 'ADA')

    # Plot
    # slopestabilityML.plot_results(accuracy_labels, accuracy_score, 'ADA_prediction')
    # slopestabilityML.plot_results(accuracy_labels_training, accuracy_score_training, 'ADA_training')

    return result_class, accuracy_score, accuracy_labels, accuracy_score_training, accuracy_labels_training, depth_estim, depth_estim_accuracy, depth_estim_labels, depth_estim_training, depth_estim_accuracy_training, depth_estim_labels_training
