#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 31.03.2021

@author: Feliks Kiszkurno
"""
# THIS IS NOT A PROPER ML METHOD

import slopestabilityML
import numpy as np
import pandas as pd


def mgc_run(test_results, random_seed):

    test_training, test_prediction = slopestabilityML.split_dataset(test_results.keys(), random_seed)

    accuracy_score_prediction = []
    accuracy_labels_prediction = []
    accuracy_score_training = []
    accuracy_labels_training = []

    # Create classifier
    clf = 0

    # Train classifier
    result_class = {}
    for test_name in test_prediction:
        classes = slopestabilityML.MGC.max_grad_classi(test_results[test_name])
        ids_temp = np.argwhere(classes >= 0.5)
        classes[ids_temp] = 1
        ids_temp = np.argwhere(classes < 0.5)
        classes[ids_temp] = 0
        result_class[test_name] = classes
        classes_correct = test_results[test_name]['CLASS'].to_numpy()
        score_prediction = len(np.argwhere(classes == classes_correct)) / len(classes_correct)
        accuracy_score_prediction.append(score_prediction * 100)
        accuracy_labels_prediction.append(test_name)
        slopestabilityML.plot_class_res(test_results, test_name, classes_correct, classes, 'MGC_prediction')


    # for test_name in test_training:
    #     classes = slopestabilityML.MGC.max_grad_classi(test_results[test_name])
    #     ids_temp = np.argwhere(classes >= 0.5)
    #     classes[ids_temp] = 1
    #     ids_temp = np.argwhere(classes < 0.5)
    #     classes[ids_temp] = 0
    #     classes_correct = test_results[test_name]['CLASS'].to_numpy()
    #     score_prediction = len(np.argwhere(classes == classes_correct)) / len(classes_correct)
    #     accuracy_score_training.append(score_prediction * 100)
    #     accuracy_labels_training.append(test_name)
    #     slopestabilityML.plot_class_res(test_results, test_name, classes_correct, classes, 'MGC_training')

    # Plot
    slopestabilityML.plot_results(accuracy_labels_prediction, accuracy_score_prediction, 'MGC_training')

    return result_class, accuracy_score_prediction, accuracy_labels_prediction, accuracy_score_training, accuracy_labels_training
