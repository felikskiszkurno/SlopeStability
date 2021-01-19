#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19.01.2021

@author: Feliks Kiszkurno
"""

import slopestabilityML
#import slopestabilityML.SVM.svm_run
#import slopestabilityML.GBC.gbc_run


def run_all_tests(test_results):

    random_seed = 999

    ml_results = {}

    print('Running SVM...')
    svm_accuracy_score, svm_accuracy_labels = slopestabilityML.SVM.svm_run(test_results, random_seed)
    ml_results['svm'] = {'score': svm_accuracy_score, 'labels': svm_accuracy_labels}

    print('Running GBC...')
    gbc_accuracy_score, gbc_accuracy_labels = slopestabilityML.GBC.gbc_run(test_results, random_seed)
    ml_results['gbc'] = {'score': gbc_accuracy_score, 'labels': gbc_accuracy_labels}

    print('Running SGD...')
    sgd_accuracy_score, sgd_accuracy_labels = slopestabilityML.SGD.sgd_run(test_results, random_seed)
    ml_results['sgd'] = {'score': sgd_accuracy_score, 'labels': sgd_accuracy_labels}

    slopestabilityML.combine_results(ml_results)
    print('end')

    return ml_results
