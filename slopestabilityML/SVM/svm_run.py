#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 16.01.2021

@author: Feliks Kiszkurno
"""

from sklearn import svm

import slopestabilityML.plot_results
import slopestabilityML.split_dataset
import slopestabilityML.run_classi


# TODO: for comparability with other ML methods, add option to define which test should be used for training externaly


def svm_run(test_results, random_seed):

    # https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/

    test_training, test_prediction = slopestabilityML.split_dataset(test_results.keys(), random_seed)

    accuracy_score = []
    accuracy_labels = []

    # Create classifier
    clf = svm.SVC(gamma=0.001, C=100, kernel='linear')

    # Train classifier
    result_class, accuracy_labels, accuracy_score, accuracy_labels_training, accuracy_score_training = \
        slopestabilityML.run_classification(test_training, test_prediction, test_results, clf, 'SVM')


    # Plot
    slopestabilityML.plot_results(accuracy_labels, accuracy_score, 'SVM_prediction')
    slopestabilityML.plot_results(accuracy_labels_training, accuracy_score_training, 'SVM_training')

    return result_class, accuracy_score, accuracy_labels, accuracy_score_training, accuracy_labels_training
