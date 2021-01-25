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
import random

# TODO: for comparability with other ML methods, add option to define which test should be used for training externaly


def svm_run(test_results, random_seed):

    # https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/

    test_training, test_prediction = slopestabilityML.split_dataset(test_results.keys(), random_seed)

    accuracy_score = []
    accuracy_labels = []

    # Create classifier
    clf = svm.SVC(gamma=0.001, C=100, kernel='linear')

    # Train classifier
    accuracy_labels, accuracy_score = slopestabilityML.run_classification(test_training, test_prediction, test_results,
                                                                          clf, 'SVM')

    # for test_name in test_training:
    #
    #     # Prepare data
    #     x_train, y_train = slopestabilityML.preprocess_data(test_results[test_name])
    #
    #     # Train classifier
    #     clf.fit(x_train, y_train)
    #
    # # Predict with classfier
    # for test_name_pred in test_prediction:
    #
    #     # Prepare data
    #     x_question, y_answer = slopestabilityML.preprocess_data(test_results[test_name_pred])
    #
    #     y_pred = clf.predict(x_question)
    #
    #     # Evaluate result
    #     accuracy_score.append(len(np.where(y_pred == y_answer)) / len(y_answer) * 100)
    #     accuracy_labels.append(test_name_pred)

    # Plot
    slopestabilityML.plot_results(accuracy_labels, accuracy_score, 'SVM')

    return accuracy_score, accuracy_labels
