#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 26.03.2021

@author: Feliks Kiszkurno
"""

import slopestabilityML.plot_results
import slopestabilityML.split_dataset
import slopestabilityML.run_classification

import numpy as np

from sklearn.ensemble import AdaBoostClassifier

#from sklearn.tree import DecisionTreeClassifier

import settings


def adaboost_run(test_results, random_seed):

    test_results, test_training, test_prediction = slopestabilityML.select_split_type(test_results, random_seed)

    if settings.settings['optimize_ml'] is True:

        hyperparameters = {'base_estimator': ['SVM', 'GBC', 'KNN'],
                      'n_estimators': list(np.arange(10, 90, 10)),
                      'learning_rate': list(np.arange(0.2, 1.5, 0.1))}

        clf_base = AdaBoostClassifier()

        clf = slopestabilityML.select_search_type(clf_base, hyperparameters)

    else:

        clf = AdaBoostClassifier()#base_estimator=DecisionTreeClassifier(max_depth=3),
                             #n_estimators=20, random_state=0)

    # Train classifier
    result_class, accuracy_labels, accuracy_score, accuracy_labels_training, accuracy_score_training, depth_estim, depth_estim_accuracy, depth_estim_labels, depth_estim_training, depth_estim_accuracy_training, depth_estim_labels_training = \
        slopestabilityML.run_classification(test_training, test_prediction, test_results, clf, 'ADA')

    # Plot
    # slopestabilityML.plot_results(accuracy_labels, accuracy_score, 'ADA_prediction')
    # slopestabilityML.plot_results(accuracy_labels_training, accuracy_score_training, 'ADA_training')

    return result_class, accuracy_score, accuracy_labels, accuracy_score_training, accuracy_labels_training, depth_estim, depth_estim_accuracy, depth_estim_labels, depth_estim_training, depth_estim_accuracy_training, depth_estim_labels_training
