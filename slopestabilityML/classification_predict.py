#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 29.05.2021

@author: Feliks Kiszkurno
"""

import settings
import slopestabilityML
import pandas as pd
import numpy as np
import os
import joblib
from scipy import interpolate

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance

import slopestabilitytools
import test_definitions


def classification_predict(test_prediction, test_results, clf_name, num_feat, *, batch_name=''):

    result_class = {}

    accuracy_result = []
    accuracy_labels = []

    depth_estim = []
    depth_true = []
    depth_estim_accuracy = []
    depth_estim_labels = []

    if settings.settings['reuse_clf'] is True:

        if clf_name in settings.settings['clf_trained']:
            clf_name_ext = clf_name + '.sav'
            clf_pipeline = joblib.load(os.path.join(settings.settings['clf_folder'], clf_name_ext))

    test_results_orig = test_results.copy()

    # Predict with classifier
    for test_name_pred in test_prediction:
        # Prepare data
        print(test_name_pred)
        # test_name_pred_orig = test_name_pred
        test_name_pred, test_name_pred_orig = slopestabilityML.check_name(test_name_pred)

        test_results = test_results_orig[test_name_pred]

        if settings.settings['min_sen_pred'] is True:
            #test_results_temp = test_results[test_name_pred]
            sen = test_results['SEN'].to_numpy()
            senn = (sen + abs(sen.min())) / (sen.max() + abs(sen.min()))
            test_results['SENN'] = senn
            test_results_temp = test_results[test_results['SENN'] > settings.settings['min_sen_pred_val']]
            x_question, y_answer, x_position = slopestabilityML.preprocess_data(test_results_temp,
                                                                                return_x=True)
        else:
            test_results_temp = test_results.copy()
            x_question, y_answer, x_position = slopestabilityML.preprocess_data(test_results,
                                                                                return_x=True)
            #x_question = x_question[num_feat]

        #weights = x_question['SEN']
        weights_np = x_question['SEN'].to_numpy()
        weights = (weights_np + abs(weights_np.min())) / (weights_np.max() + abs(weights_np.min()))

        if settings.settings['weight'] is True:
            x_question = x_question[num_feat]
            try:
                y_pred = clf_pipeline.predict(x_question, **{clf_pipeline.steps[1][0]+'__sample_weight': weights})
            except TypeError:
                x_question = x_question[num_feat]
                y_pred = clf_pipeline.predict(x_question)
        else:
            x_question = x_question[num_feat]
            y_pred = clf_pipeline.predict(x_question)

        result_class[test_name_pred] = y_pred
        # print(y_pred)
        score = accuracy_score(y_answer, y_pred)
        # print('{bn}, {tn} score: {score:.2f} %'.format(bn=batch_name, tn=test_name_pred, score=score * 100))

        slopestabilityML.plot_sen_corr(y_pred, y_answer.to_numpy().reshape(y_answer.size), weights_np,
                                       clf_name, test_name_pred, batch_name,
                                       training=False)

        importance = permutation_importance(clf_pipeline, x_question, y_pred)

        slopestabilityML.plot_feature_importance(clf_pipeline, importance, x_question, test_name_pred,
                                                 batch_name=batch_name)

        log_file_name = settings.settings['log_file_name']
        log_file = open(os.path.join(settings.settings['results_folder'], log_file_name), 'a')
        log_file.write('\n')
        log_file.write('{bn}, {tn} score: {score:.2f} %'.format(bn=batch_name, tn=test_name_pred, score=score * 100))
        log_file.write('\n')
        log_file.write('{bn}, {tn} feature list: {fl}'.format(bn=batch_name, tn=test_name_pred,
                                                              fl=x_question.columns.values.tolist()))
        log_file.write('\n')
        log_file.write('{bn}, {tn}  feature importance: {fi}'.format(bn=batch_name, tn=test_name_pred,
                                                                     fi=importance.importances_mean))
        log_file.write('\n')
        log_file.close()

        if settings.settings['norm_class'] is True:
            class_in = test_results_temp['CLASSN']
        elif settings.settings['norm_class'] is False and settings.settings['use_labels'] is False:
            class_in = test_results_temp['CLASS']
        elif settings.settings['use_labels'] is True:
            class_in = test_results_temp['LABELS']
        else:
            print('I don\'t know which class to use! Exiting...')
            exit(0)

        # Evaluate the accuracy of interface depth detection
        x = x_position.to_numpy()
        x = x.reshape([x.size])
        y = test_results_temp['Y'].to_numpy()
        xi, yi, gridded_data = slopestabilitytools.grid_data(x, y, {'class': y_pred})
        y_pred_grid = gridded_data['class']

        result_grid_rolled = np.roll(y_pred_grid, -1, axis=0)
        y_pred_grid_deri = y_pred_grid - result_grid_rolled
        y_pred_grid_deri[-1, :] = 0
        interfaces_detected = slopestabilitytools.detect_interface(xi, yi, y_pred_grid)
        depth_interface_estimate = {}
        y_estimate_interp = {}
        x_interpolator = {}
        depth_interface_estimate_count = 0
        depth_interface_accuracy_mean = 0
        depth_interface_estimate_mean = 0
        depth_interface_estimate_count = 0
        depth_interface_true = test_definitions.test_parameters[test_name_pred]['layers_pos']
        depth_interface_accuracy = 0
        depth_detected = []
        depth_detected_true = []
        for interfaces_key in interfaces_detected.keys():

            diff = abs(np.ones([len(depth_interface_true)]) * interfaces_detected[interfaces_key][
                'depth_mean'] - depth_interface_true)
            error_file = open(os.path.join(settings.settings['results_folder'], 'error_file.txt'), 'a')
            error_file.write(test_name_pred)
            error_file.write('\n')
            # print(np.array2string(depth_interface_true))
            error_file.write(np.array2string(depth_interface_true))
            error_file.write('\n')
            # print(interfaces_detected[interfaces_key])
            error_file.write(str(interfaces_detected[interfaces_key]))
            error_file.write('\n')
            # print(np.array2string(diff))
            error_file.write(np.array2string(diff))
            error_file.write('\n')
            best_match_id = np.argwhere(diff == np.min(diff))
            error_file.write(np.array2string(best_match_id))
            error_file.write('\n')
            # print(best_match_id)
            best_match_depth = depth_interface_true[best_match_id[0]]
            error_file.write(np.array2string(best_match_depth))
            error_file.write('\n')
            error_file.close()
            depth_interface_estimate[interfaces_key] = interfaces_detected[interfaces_key]['depth_mean']
            depth_detected.append(interfaces_detected[interfaces_key]['depth_mean'])
            depth_detected_true.append(best_match_depth[0])
            # depth_interface_estimate_mean = depth_interface_estimate_mean + interfaces_detected[interfaces_key]['depth_mean']
            # depth_interface_estimate_count += 1
            y_estimate = interfaces_detected[interfaces_key]['depths']
            x_estimate = interfaces_detected[interfaces_key]['x']
            # depth_interface_accuracy = ((depth_interface_estimate-test_definitions.test_parameters[name]['layers_pos'][0])/test_definitions.test_parameters[name]['layers_pos'][0])*100
            y_actual = np.ones([y_estimate.size]) * \
                       best_match_depth
            y_actual = y_actual.reshape([y_actual.shape[0]])
            depth_interface_accuracy = mean_squared_error(y_actual[np.isfinite(y_estimate)],
                                                          y_estimate[np.isfinite(y_estimate)],
                                                          squared=False)
            depth_interface_accuracy = (depth_interface_accuracy / abs(best_match_depth[0])) * 100
            depth_interface_accuracy_mean += depth_interface_accuracy
            depth_interface_estimate_count += 1
            interpolator = interpolate.interp1d(x_estimate[np.isfinite(y_estimate)],
                                                y_estimate[np.isfinite(y_estimate)])  # bounds_error=False,
                                                # fill_value='extrapolate')
            x_interpolator[interfaces_key] = sorted(x[(x > x_estimate[np.isfinite(y_estimate)].min()) & (x < x_estimate[np.isfinite(y_estimate)].max())])
            y_estimate_interp[interfaces_key] = interpolator(x_interpolator[interfaces_key])

        if depth_interface_estimate_count == 0:
            depth_interface_accuracy_mean = 0
        else:
            depth_interface_accuracy_mean = depth_interface_accuracy_mean / depth_interface_estimate_count
        # depth_estim.append(depth_interface_estimate_mean/depth_interface_estimate_count)
        depth_estim.append(depth_detected)
        depth_true.append(depth_detected_true)
        depth_estim_accuracy.append(depth_interface_accuracy_mean)
        depth_estim_labels.append(
            test_name_pred + '_' + str(test_definitions.test_parameters[test_name_pred]['layers_pos'][0]))

        # slopestabilityML.plot_class_overview(test_results[test_name_pred], test_name_pred, class_in, y_pred, clf_name, depth_estimate=depth_interface_estimate,
        #                                     depth_accuracy=depth_interface_accuracy)

        slopestabilityML.plot_class_overview(test_results_temp, test_name_pred_orig, class_in, y_pred,
                                             clf_name, training=False, depth_estimate=depth_interface_estimate,
                                             interface_y=y_estimate_interp, interface_x=x_interpolator,
                                             depth_accuracy=depth_interface_accuracy_mean, batch_name=batch_name)

        # Evaluate result
        # accuracy_.append(len(np.where(y_pred == y_answer.to_numpy())) / len(y_answer.to_numpy()) * 100)
        accuracy_result.append(score * 100)
        accuracy_labels.append(test_name_pred_orig)

        # Evaluate

    return result_class, accuracy_labels, accuracy_result, depth_estim, depth_true, \
        depth_estim_accuracy, depth_estim_labels,
