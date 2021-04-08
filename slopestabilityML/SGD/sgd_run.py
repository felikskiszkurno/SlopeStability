#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19.01.2021

@author: Feliks Kiszkurno
"""

from sklearn.linear_model import SGDClassifier

import slopestabilityML.plot_results
import slopestabilityML.split_dataset
import slopestabilityML.run_classi


def sgd_run(test_results, random_seed):

    test_training, test_prediction = slopestabilityML.split_dataset(test_results.keys(), random_seed)

    # Create classifier
    clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)

    # Train classifier
    result_class, accuracy_labels, accuracy_score, accuracy_labels_training, accuracy_score_training = \
        slopestabilityML.run_classification(test_training, test_prediction, test_results, clf, 'SGD')

    # Plot
    slopestabilityML.plot_results(accuracy_labels, accuracy_score, 'SGD_prediction')
    slopestabilityML.plot_results(accuracy_labels_training, accuracy_score_training, 'SGD_training')

    return result_class, accuracy_score, accuracy_labels, accuracy_score_training, accuracy_labels_training