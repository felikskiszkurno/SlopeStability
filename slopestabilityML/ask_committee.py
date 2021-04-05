#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 04.04.2021

@author: Feliks Kiszkurno
"""

import numpy as np

import settings
import slopestabilityML


def ask_committee(ml_result_class, test_results, *, random_seed=False):
    
    results = np.array([])
    classes_correct = {}
    for test_name in sorted(test_results.keys()):
        if settings.settings['norm_class'] is True:
            class_in = test_results[test_name]['CLASSN']
        elif settings.settings['norm_class'] is False:
            class_in = test_results[test_name]['CLASS']
        else:
            print('I don\'t know which class to use! Exiting...')
            exit(0)
        classes_correct[test_name] = class_in
        
    # First create list of datasets and verify if all classifiers has been run over the same sets
    test_names_all = []
    for method_name in sorted(ml_result_class.keys()):
        test_names_all_temp = []
        for test_name in sorted(ml_result_class[method_name].keys()):
            test_names_all_temp.append(test_name)
        test_names_all = set(test_names_all_temp)
        #if sorted(test_names_all) == sorted(test_names_all_temp):
        #    test_names_all = sorted(test_names_all_temp)
        #else:
        #    print("Mismatch in datasets!")
        #    exit(0)

    # Combine results for each data set from each classifier into an array
    results_test = {}
    for test_name in test_names_all:
        results = np.zeros([len(ml_result_class[list(ml_result_class.keys())[0]][test_name]), len( sorted(ml_result_class.keys()))])
        method_id = 0
        for method_name in sorted(ml_result_class.keys()):
            results[:, method_id] = ml_result_class[method_name][test_name].T.reshape([len(ml_result_class[method_name][test_name])])
            method_id = method_id + 1
        results_test[test_name] = results

    # Execute the voting
    results_voting = {}
    for test_name in sorted(results_test.keys()):
        test_result = results_test[test_name]
        shape_test = test_result.shape
        result_rows = shape_test[0]
        result_cols = shape_test[1]
        result_voted = np.zeros([result_rows, 1])
        for row_id in range(result_rows):
            row_temp = test_result[row_id, :]
            result_voted[row_id] = np.bincount(row_temp.astype(int)).argmax()
        results_voting[test_name] = result_voted

    # Compare voted result with correct result
    accuracy_score = []
    accuracy_labels = []
    accuracy_score_training = []
    accuracy_labels_training = []

    # if random_seed is not False:
    #     test_training, test_prediction = slopestabilityML.split_dataset(test_results.keys(), random_seed)
    #     tests_ordered = {'train': test_training, 'pred': test_prediction}
    # else:
    test_training = []
    test_prediction = results_voting.keys()
    tests_ordered = {'train': test_training, 'pred': test_prediction}

    for test_group in sorted(tests_ordered.keys()):
        for test_name in tests_ordered[test_group]:
            classes_correct = class_in.to_numpy()
            classes_correct = classes_correct.reshape([len(results_voting[test_name]), 1])
            score = len(np.argwhere(results_voting[test_name] == classes_correct)) / len(classes_correct)
            if test_group is 'train':
                accuracy_score_training.append(score * 100)
                accuracy_labels_training.append(test_name)
            elif test_group is 'pred':
                accuracy_score.append(score * 100)
                accuracy_labels.append(test_name)

    return accuracy_score, accuracy_labels, accuracy_score_training, accuracy_labels_training
