#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 26.01.2021

@author: Feliks Kiszkurno
"""

from sklearn.neighbors import KNeighborsClassifier
import slopestabilityML.plot_results
import slopestabilityML.split_dataset
import slopestabilityML.run_classi

def knn_run(test_results, random_seed):

    # Split the data set
    test_training, test_prediction = slopestabilityML.split_dataset(test_results.keys(), random_seed)

    # Create classifier
    clf = KNeighborsClassifier(n_neighbors=2)

    # Train classifier
    result_class, accuracy_labels, accuracy_score, accuracy_labels_training, accuracy_score_training = \
        slopestabilityML.run_classification(test_training, test_prediction, test_results, clf, 'KNN')

    # Plot
    slopestabilityML.plot_results(accuracy_labels, accuracy_score, 'KNN_prediction')
    slopestabilityML.plot_results(accuracy_labels_training, accuracy_score_training, 'KNN_training')

    return result_class, accuracy_score, accuracy_labels, accuracy_score_training, accuracy_labels_training