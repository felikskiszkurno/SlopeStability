#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19.01.2021

@author: Feliks Kiszkurno
"""

from sklearn import ensemble

import numpy as np

import slopestabilityML.plot_results
import slopestabilityML.split_dataset
import slopestabilityML.run_classification

import settings


def gbc_run(test_results, random_seed):

    # Split the data set
    test_results, test_training, test_prediction = slopestabilityML.select_split_type(test_results, random_seed)

    if settings.settings['optimize_ml'] is True:
        
        hyperparameters = {'loss': ['deviance', 'exponential'],
                           'learning_rate': list(np.arange(0.2, 1.5, 0.1)),
                           'n_estimators': list(np.arange(70, 160, 10))}

        clf_base = ensemble.GradientBoostingClassifier()

        clf = slopestabilityML.select_search_type(clf_base, hyperparameters)

    else:
        # Create classifier
        clf = ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

    # Train classifier
    results, result_class = \
        slopestabilityML.run_classification(test_training, test_prediction, test_results, clf, 'GBC')

    # Plot
    # slopestabilityML.plot_results(accuracy_labels, accuracy_score, 'GBC_prediction')
    # slopestabilityML.plot_results(accuracy_labels_training, accuracy_score_training, 'GBC_training')

    return results, result_class
