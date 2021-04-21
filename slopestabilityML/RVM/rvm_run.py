#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 31.03.2021

@author: Feliks Kiszkurno
"""

from sklearn_rvm import EMRVC

import slopestabilityML.plot_results
import slopestabilityML.split_dataset
import slopestabilityML.run_classification


def rvm_run(test_results, random_seed):
    test_training, test_prediction = slopestabilityML.split_dataset(test_results.keys(), random_seed)

    accuracy_score = []
    accuracy_labels = []

    # Create classifier
    clf = EMRVC(kernel="rbf")

    # Train classifier
    accuracy_labels, accuracy_score, accuracy_labels_training, accuracy_score_training = \
        slopestabilityML.run_classification(test_training, test_prediction, test_results, clf, 'RVM')

    # Plot
    # slopestabilityML.plot_results(accuracy_labels, accuracy_score, 'RVM_prediction')
    # slopestabilityML.plot_results(accuracy_labels_training, accuracy_score_training, 'RVM_training')

    return accuracy_score, accuracy_labels, accuracy_score_training, accuracy_labels_training
