#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 16.01.2021

@author: Feliks Kiszkurno
"""

from sklearn import svm

import slopestabilityML.plot_results
import slopestabilityML.split_dataset
import slopestabilityML.run_classification
import settings


def svm_run(test_results, random_seed):

    # https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/

    # Split the data set
    if settings.settings['data_split'] is 'random':
        test_training, test_prediction = slopestabilityML.split_dataset(test_results.keys(), random_seed)
        test_results_mixed = test_results
    elif settings.settings['data_split'] is 'predefined':
        test_training = test_results['training'].keys()
        test_prediction = test_results['prediction'].keys()
        test_results_mixed = {}
        test_results_mixed.update(test_results['prediction'])
        test_results_mixed.update(test_results['training'])

    accuracy_score = []
    accuracy_labels = []

    # Create classifier
    clf = svm.SVC(gamma=0.001, C=100, kernel='linear')

    # Train classifier
    result_class, accuracy_labels, accuracy_score, accuracy_labels_training, accuracy_score_training, depth_estim, depth_estim_accuracy, depth_estim_labels, depth_estim_training, depth_estim_accuracy_training, depth_estim_labels_training = \
        slopestabilityML.run_classification(test_training, test_prediction, test_results_mixed, clf, 'SVM')


    # Plot
    # slopestabilityML.plot_results(accuracy_labels, accuracy_score, 'SVM_prediction')
    # slopestabilityML.plot_results(accuracy_labels_training, accuracy_score_training, 'SVM_training')

    return result_class, accuracy_score, accuracy_labels, accuracy_score_training, accuracy_labels_training, depth_estim, depth_estim_accuracy, depth_estim_labels, depth_estim_training, depth_estim_accuracy_training, depth_estim_labels_training
