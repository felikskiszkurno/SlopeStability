#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 26.03.2021

@author: Feliks Kiszkurno
"""

import slopestabilityML.plot_results
import slopestabilityML.split_dataset
import slopestabilityML.run_classi

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


def adaboost_run(test_results, random_seed):
    test_training, test_prediction = slopestabilityML.split_dataset(test_results.keys(), random_seed)

    clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3),
                             n_estimators=20, random_state=0)

    # Train classifier
    result_class, accuracy_labels, accuracy_score, accuracy_labels_training, accuracy_score_training = \
        slopestabilityML.run_classification(test_training, test_prediction, test_results, clf, 'ADA')

    # Plot
    slopestabilityML.plot_results(accuracy_labels, accuracy_score, 'ADA_prediction')
    slopestabilityML.plot_results(accuracy_labels_training, accuracy_score_training, 'ADA_training')

    return result_class, accuracy_score, accuracy_labels, accuracy_score_training, accuracy_labels_training
