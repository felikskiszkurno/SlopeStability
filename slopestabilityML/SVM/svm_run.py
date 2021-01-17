#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 16.01.2021

@author: Feliks Kiszkurno
"""

from sklearn import svm
import slopestabilitytools
import random
import math
import numpy as np
import slopestabilityML.plot_results

# TODO: for comparability with other ML methods, add option to define which test should be used for training externaly

def svm_run(test_results):
    # https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/

    accuracy_score = []
    accuracy_labels = []

    test_number = len(test_results.keys())
    test_prediction = random.choices(list(test_results.keys()),
                                     k=math.ceil(test_number * 0.1))

    test_training = slopestabilitytools.set_diff(list(test_results.keys()), set(test_prediction))
    print(test_prediction)
    # Create classifier
    clf = svm.SVC(gamma=0.001, C=100, kernel='linear')

    # Train classifier

    for test_name in test_training:
        # Prepare data
        data_set = test_results[test_name]
        x_train = data_set.drop(['Z', 'INM', 'CLASS'], axis='columns')
        y_train = data_set['CLASS']

        # Train classifier
        clf.fit(x_train, y_train)

    # Predict with classfier
    for test_name_pred in test_prediction:
        # Prepare data

        data_set_pred = test_results[test_name_pred]
        print(data_set_pred)
        x_question = data_set_pred.drop(['Z', 'INM', 'CLASS'], axis='columns')
        y_answer = data_set_pred['CLASS']

        y_pred = clf.predict(x_question)

        # Evaluate result
        accuracy_score.append(len(np.where(y_pred == y_answer)) / len(y_answer) * 100)
        accuracy_labels.append(test_name_pred)

    # Plot
    slopestabilityML.plot_results(accuracy_labels, accuracy_score)

    return accuracy_score, accuracy_labels
