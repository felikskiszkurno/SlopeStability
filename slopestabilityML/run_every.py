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
import slopestabilityML.DNN.dnn_run
import slopestabilityML
import os
import settings

import gc


def run_all_tests(test_results):

    error_file = open(os.path.join(settings.settings['results_folder'], 'error_file.txt'), 'w')

    random_seed = 999

    ml_results = {}
    ml_results_class = {}

    print('Running SVM...')
    svm_results, svm_result_class = slopestabilityML.SVM.svm_run(test_results, random_seed)
    ml_results['svm'] = svm_results

    ml_results_class['svm'] = svm_result_class

    gc.collect()

    print('Running GBC...')
    gbc_results, gbc_result_class = slopestabilityML.GBC.gbc_run(test_results, random_seed)
    ml_results['gbc'] = gbc_results

    ml_results_class['gbc'] = gbc_result_class

    gc.collect()

    print('Running SGD...')
    sgd_results, sgd_result_class = slopestabilityML.SGD.sgd_run(test_results, random_seed)
    ml_results['sgd'] = sgd_results

    ml_results_class['sgd'] = sgd_result_class

    gc.collect()

    print('Running KNN...')
    knn_results, knn_result_class = slopestabilityML.KNN.knn_run(test_results, random_seed)
    ml_results['KNN'] = knn_results

    ml_results_class['knn'] = knn_result_class

    gc.collect()

    print('Running ADABOOST...')
    ada_results, ada_result_class = slopestabilityML.ADABOOST.adaboost_run(test_results, random_seed)
    ml_results['ADA'] = ada_results

    ml_results_class['ada'] = ada_result_class

    gc.collect()

    print('Running DNN...')
    ada_results, ada_result_class = slopestabilityML.DNN.dnn_run(test_results, random_seed)
    ml_results['DNN'] = ada_results

    ml_results_class['dnn'] = ada_result_class

    gc.collect()

    # ada_result_class, ada_accuracy_score, ada_accuracy_labels, ada_accuracy_score_training, ada_accuracy_labels_training, ada_depth_estim, ada_depth_estim_accuracy, ada_depth_estim_labels, ada_depth_estim_training, ada_depth_estim_accuracy_training, ada_depth_estim_labels_training = \
    #     slopestabilityML.ADABOOST.adaboost_run(test_results, random_seed)
    # ml_results['ADABOOST'] = {'score': ada_accuracy_score, 'labels': ada_accuracy_labels,
    #                      'score_training': ada_accuracy_score_training, 'labels_training': ada_accuracy_labels_training,
    #                      'depth_estim': ada_depth_estim, 'depth_accuracy': ada_depth_estim_accuracy,
    #                      'depth_labels': ada_depth_estim_labels,
    #                      'depth_estim_training': ada_depth_estim_training,
    #                      'depth_accuracy_training': ada_depth_estim_accuracy_training,
    #                      'depth_labels_training': ada_depth_estim_labels_training
    #                      }
    # ml_results_class['ADABOOST'] = ada_result_class
    # gc.collect()

    # print('Running RVM...')
    # rvm_accuracy_score, rvm_accuracy_labels, rvm_accuracy_score_training, rvm_accuracy_labels_training = \
    #     slopestabilityML.RVM.rvm_run(test_results, random_seed)
    # ml_results['RVM'] = {'score': rvm_accuracy_score, 'labels': rvm_accuracy_labels,
    #                      'score_training': rvm_accuracy_score_training, 'labels_training': rvm_accuracy_labels_training}

    # print('Running MGC')
    # mgc_result_class, mgc_accuracy_score, mgc_accuracy_labels, mgc_accuracy_score_training, mgc_accuracy_labels_training \
    #     = slopestabilityML.MGC.mgc_run(test_results, random_seed)
    # ml_results['MGC'] = {'score': mgc_accuracy_score, 'labels': mgc_accuracy_labels,
    #                      'score_training': mgc_accuracy_score_training, 'labels_training': mgc_accuracy_labels_training}
    # ml_results_class['mgc'] = mgc_result_class

    # Here should be called function, that will plot true vs predicted depth plot

    print('Asking committee for verdict')

    if settings.settings['use_batches'] is False:

        results_com \
            = slopestabilityML.ask_committee(ml_results_class, test_results, random_seed=random_seed)
        #ml_results['com'] = {'score': committee_accuracy_score, 'labels': committee_accuracy_labels,
        #                     'score_training': committee_accuracy_score_training,
        #                     'labels_training': committee_accuracy_labels_training}
        ml_results['com'] = results_com

        del results_com

        slopestabilityML.combine_results(ml_results)
        slopestabilityML.plot_depth_true_estim(ml_results)

    elif settings.settings['use_batches'] is True:

        ml_results['com'] = {}
        ml_results['com']['training'] = {}
        ml_results['com']['prediction'] = {}
        ml_result_com_batches = {}

        for batch_name in test_results['prediction'].keys():

            test_results_temp = {}
            test_results_temp['training'] = test_results['training']
            test_results_temp['prediction'] = test_results['prediction'][batch_name]

            ml_results_class_temp = {}
            for method_name in ml_results_class.keys():
                ml_results_class_temp[method_name] = ml_results_class[method_name][batch_name]

            results_com \
                = slopestabilityML.ask_committee(ml_results_class_temp, test_results_temp, random_seed=random_seed)
            # ml_results['com'] = {'score': committee_accuracy_score, 'labels': committee_accuracy_labels,
            #                     'score_training': committee_accuracy_score_training,
            #                     'labels_training': committee_accuracy_labels_training}
            ml_results['com']['training'][batch_name] = results_com['training']
            ml_results['com']['prediction'][batch_name] = results_com['prediction']
            slopestabilityML.combine_results(ml_results, batch_name=batch_name)
            slopestabilityML.plot_depth_true_estim(ml_results, batch_name=batch_name)
            #ml_results.pop('com')
            #ml_results['com'][batch_name] = resu

    gc.collect()
    print('ML classification finished')

    return ml_results
