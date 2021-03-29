#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19.01.2021

@author: Feliks Kiszkurno
"""

from sklearn import ensemble
import slopestabilityML.plot_results
import slopestabilityML.split_dataset
import slopestabilityML.run_classi

# TODO: as in svm_run


def gbc_run(test_results, random_seed):

    test_training, test_prediction = slopestabilityML.split_dataset(test_results.keys(), random_seed)

    # Create classifier
    clf = ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

    # Train classifier
    accuracy_labels, accuracy_score, accuracy_labels_training, accuracy_score_training = \
        slopestabilityML.run_classification(test_training, test_prediction, test_results, clf, 'GBC')

    # Plot
    slopestabilityML.plot_results(accuracy_labels, accuracy_score, 'GBC_prediction')
    slopestabilityML.plot_results(accuracy_labels_training, accuracy_score_training, 'GBC_training')

    return accuracy_score, accuracy_labels, accuracy_score_training, accuracy_labels_training
