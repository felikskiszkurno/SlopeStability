#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19.01.2021

@author: Feliks Kiszkurno
"""
import joblib
import os

import settings
import slopestabilityML

import slopestabilitytools


def run_classification(test_training, test_prediction, test_results, clf, clf_name, *, hyperparameters=False, batch_name=''):

    result_training = {}

    clf_result_file_ext = clf_name + '_result.sav'
    clf_result_file = os.path.join(settings.settings['clf_folder'], clf_result_file_ext)

    if clf_name not in settings.settings['clf_trained']:
        result_class_training, depth_estim_training, depth_true_training, depth_estim_accuracy_training, \
        depth_estim_labels_training, accuracy_score_training, accuracy_labels_training = \
            slopestabilityML.classification_train(test_training, test_results, clf, clf_name)

        result_training = {'result_class': result_class_training,
                           'accuracy_score': accuracy_score_training,
                           'accuracy_labels': accuracy_labels_training,
                           'depth_estim': depth_estim_training,
                           'depth_true': depth_true_training,
                           'depth_estim_accuracy': depth_estim_accuracy_training,
                           'depth_estim_labels': depth_estim_labels_training}

        settings.settings['clf_trained'].append(clf_name)

        joblib.dump(result_training, clf_result_file)

    else:

        result_training = joblib.load(clf_result_file)

        #if settings.settings['retrain_clf'] is False:
        #    settings.settings['clf_trained'] = True

    # if settings.settings['clf_trained'] is True & settings.settings['clf_trained'] is True:
    #     clf_file_name = os.path.join(settings.settings['clf_folder'], clf_name, '.sav')
    #     clf = joblib.load(clf_file_name)

    result_prediction = {}

    if settings.settings['use_batches'] is True:
        for batch in test_prediction:

            slopestabilitytools.folder_structure.create_folder_structure(batch_names=[batch])

            log_file_name = batch_name + '_log.txt'
            log_file = open(os.path.join(settings.settings['figures_folder'], log_file_name), 'w')
            log_file.write('Starting log file for: ' + batch_name)
            log_file.close()

            result_class, accuracy_labels, accuracy_result, depth_estim, depth_true, \
            depth_estim_accuracy, depth_estim_labels = \
                slopestabilityML.classification_predict(test_prediction[batch], test_results, clf_name, batch_name=batch)

            result_prediction[batch] = {'result_class': result_class,
                                        'accuracy_score': accuracy_result,
                                        'accuracy_labels': accuracy_labels,
                                        'depth_estim': depth_estim,
                                        'depth_true': depth_true,
                                        'depth_estim_accuracy': depth_estim_accuracy,
                                        'depth_estim_labels': depth_estim_labels,
                                        }

    else:
        result_class, accuracy_labels, accuracy_result, depth_estim, depth_true, \
        depth_estim_accuracy, depth_estim_labels = \
            slopestabilityML.classification_train(test_prediction, test_results)

        result_prediction['no_batch'] = {'result_class': result_class,
                                         'accuracy_labels': accuracy_labels,
                                         'accuracy_score': accuracy_result,
                                         'depth_estim': depth_estim,
                                         'depth_true': depth_true,
                                         'depth_estim_accuracy': depth_estim_accuracy,
                                         'depth_estim_labels': depth_estim_labels,
                                         }

    results = {'training': result_training,
               'prediction': result_prediction}

    return results, result_class
