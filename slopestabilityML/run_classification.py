#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19.01.2021

@author: Feliks Kiszkurno
"""

import slopestabilityML
import numpy as np


def run_classification(test_training, test_prediction, test_results, clf):

    accuracy_score = []
    accuracy_labels = []

    for test_name in test_training:
        # Prepare data
        x_train, y_train = slopestabilityML.preprocess_data(test_results[test_name])

        # Train classifier
        clf.fit(x_train, y_train)

    # Predict with classifier
    for test_name_pred in test_prediction:
        # Prepare data
        x_question, y_answer = slopestabilityML.preprocess_data(test_results[test_name_pred])

        y_pred = clf.predict(x_question)

        # Evaluate result
        accuracy_score.append(len(np.where(y_pred == y_answer)) / len(y_answer) * 100)
        accuracy_labels.append(test_name_pred)

    return accuracy_labels, accuracy_score

