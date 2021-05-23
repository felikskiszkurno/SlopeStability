#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 16.01.2021

@author: Feliks Kiszkurno
"""

from sklearn import svm

import numpy as np

import slopestabilityML.plot_results
import slopestabilityML.split_dataset
import slopestabilityML.run_classification
import settings


def svm_run(test_results, random_seed):

    # https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/

    # Split the data set
    test_results, test_training, test_prediction = slopestabilityML.select_split_type(test_results, random_seed)

    accuracy_score = []
    accuracy_labels = []

    if settings.settings['optimize_ml'] is True:

        hyperparameters = {'C': list(np.argmin(0.5, 1.5, 0.1)),
                           'kernel': ['linear', 'rbf', 'poly', 'sigmoid', 'precomputed'],
                           'decision_function_shape': ['ovr', 'ovo']}

        clf_base = svm.SVC()

        clf = slopestabilityML.select_search_type(clf_base, hyperparameters)

    else:

        # Create classifier
        clf = svm.SVC(gamma=0.001, C=100, kernel='linear')

    # Train classifier
    result_class, accuracy_labels, accuracy_score, accuracy_labels_training, accuracy_score_training, depth_estim, depth_estim_accuracy, depth_estim_labels, depth_estim_training, depth_estim_accuracy_training, depth_estim_labels_training = \
        slopestabilityML.run_classification(test_training, test_prediction, test_results, clf, 'SVM')


    # Plot
    # slopestabilityML.plot_results(accuracy_labels, accuracy_score, 'SVM_prediction')
    # slopestabilityML.plot_results(accuracy_labels_training, accuracy_score_training, 'SVM_training')

    return result_class, accuracy_score, accuracy_labels, accuracy_score_training, accuracy_labels_training, depth_estim, depth_estim_accuracy, depth_estim_labels, depth_estim_training, depth_estim_accuracy_training, depth_estim_labels_training
