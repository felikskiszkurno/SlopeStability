#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19.01.2021

@author: Feliks Kiszkurno
"""

# import slopestabilityML.combine_results
import slopestabilityML.SVM.svm_run
import slopestabilityML.GBC.gbc_run
import slopestabilityML.SGD.sgd_run
import slopestabilityML.KNN.knn_run
import slopestabilityML.ADABOOST.adaboost_run
import slopestabilityML


def run_all_tests(test_results):
    random_seed = 999

    ml_results = {}

    print('Running SVM...')
    svm_accuracy_score, svm_accuracy_labels, svm_accuracy_score_training, svm_accuracy_labels_training = \
        slopestabilityML.SVM.svm_run(test_results, random_seed)
    ml_results['svm'] = {'score': svm_accuracy_score, 'labels': svm_accuracy_labels,
                         'score_training': svm_accuracy_score_training, 'labels_training': svm_accuracy_labels_training}

    print('Running GBC...')
    gbc_accuracy_score, gbc_accuracy_labels, gbc_accuracy_score_training, gbc_accuracy_labels_training = \
        slopestabilityML.GBC.gbc_run(test_results, random_seed)
    ml_results['gbc'] = {'score': gbc_accuracy_score, 'labels': gbc_accuracy_labels,
                         'score_training': gbc_accuracy_score_training, 'labels_training': gbc_accuracy_labels_training}

    print('Running SGD...')
    sgd_accuracy_score, sgd_accuracy_labels, sgd_accuracy_score_training, sgd_accuracy_labels_training = \
        slopestabilityML.SGD.sgd_run(test_results, random_seed)
    ml_results['sgd'] = {'score': sgd_accuracy_score, 'labels': sgd_accuracy_labels,
                         'score_training': sgd_accuracy_score_training, 'labels_training': sgd_accuracy_labels_training}

    print('Running KNN...')
    knn_accuracy_score, knn_accuracy_labels, knn_accuracy_score_training, knn_accuracy_labels_training = \
        slopestabilityML.KNN.knn_run(test_results, random_seed)
    ml_results['KNN'] = {'score': knn_accuracy_score, 'labels': knn_accuracy_labels,
                         'score_training': knn_accuracy_score_training, 'labels_training': knn_accuracy_labels_training}

    print('Running ADABOOST...')
    ada_accuracy_score, ada_accuracy_labels, ada_accuracy_score_training, ada_accuracy_labels_training = \
        slopestabilityML.ADABOOST.adaboost_run(test_results, random_seed)
    ml_results['ADA'] = {'score': ada_accuracy_score, 'labels': ada_accuracy_labels,
                         'score_training': ada_accuracy_score_training, 'labels_training': ada_accuracy_labels_training}

    # print('Running RVM...')
    # rvm_accuracy_score, rvm_accuracy_labels, rvm_accuracy_score_training, rvm_accuracy_labels_training = \
    #     slopestabilityML.RVM.rvm_run(test_results, random_seed)
    # ml_results['RVM'] = {'score': rvm_accuracy_score, 'labels': rvm_accuracy_labels,
    #                      'score_training': rvm_accuracy_score_training, 'labels_training': rvm_accuracy_labels_training}

    print('Running MGC')
    mgc_accuracy_score, mgc_accuracy_labels = slopestabilityML.MGC.mgc_run(test_results, random_seed)
    ml_results['MGC'] = {'score': mgc_accuracy_score, 'labels': mgc_accuracy_labels,
                         'score_training': [], 'labels_training': []}

    slopestabilityML.combine_results(ml_results)
    print('ML classification finished')

    return ml_results
