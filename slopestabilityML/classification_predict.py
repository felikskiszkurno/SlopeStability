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

import slopestabilitytools
import test_definitions


def classification_predict(test_prediction, test_results, clf_name, *, batch_name=''):

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

    # Predict with classifier
    for test_name_pred in test_prediction:
        # Prepare data
        print(test_name_pred)
        # test_name_pred_orig = test_name_pred
        test_name_pred, test_name_pred_orig = slopestabilityML.check_name(test_name_pred)
        x_question, y_answer, x_position = slopestabilityML.preprocess_data(test_results[test_name_pred], return_x=True)
        y_pred = clf_pipeline.predict(x_question)
        result_class[test_name_pred] = y_pred
        # print(y_pred)
        score = accuracy_score(y_answer, y_pred)
        print('score: {score:.2f} %'.format(score=score * 100))

        if settings.settings['norm_class'] is True:
            class_in = test_results[test_name_pred]['CLASSN']
        elif settings.settings['norm_class'] is False and settings.settings['use_labels'] is False:
            class_in = test_results[test_name_pred]['CLASS']
        elif settings.settings['use_labels'] is True:
            class_in = test_results[test_name_pred]['LABELS']
        else:
            print('I don\'t know which class to use! Exiting...')
            exit(0)

        # Evaluate the accuracy of interface depth detection
        x = x_position.to_numpy()
        x = x.reshape([x.size])
        y = x_question['Y'].to_numpy()
        xi, yi, gridded_data = slopestabilitytools.grid_data(x, y, {'class': y_pred})
        y_pred_grid = gridded_data['class']

        result_grid_rolled = np.roll(y_pred_grid, -1, axis=0)
        y_pred_grid_deri = y_pred_grid - result_grid_rolled
        y_pred_grid_deri[-1, :] = 0
        interfaces_detected = slopestabilitytools.detect_interface(xi, yi, y_pred_grid)
        depth_interface_estimate = {}
        y_estimate_interp = {}
        depth_interface_estimate_count = 0
        depth_interface_accuracy_mean = 0
        depth_interface_estimate_mean = 0
        depth_interface_estimate_count = 0
        depth_interface_true = test_definitions.test_parameters[test_name_pred]['layers_pos']
        depth_detected = []
        depth_detected_true = []
        for interfaces_key in interfaces_detected.keys():
            diff = abs(np.ones([len(depth_interface_true)]) * interfaces_detected[interfaces_key][
                'depth_mean'] - depth_interface_true)
            best_match_id = np.argwhere(diff == np.min(diff))
            best_match_depth = depth_interface_true[best_match_id[0]]
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
                                                          y_estimate[np.isfinite(y_estimate)], squared=False)
            depth_interface_accuracy_mean += depth_interface_accuracy
            interpolator = interpolate.interp1d(x_estimate[np.isfinite(y_estimate)],
                                                y_estimate[np.isfinite(y_estimate)],  # bounds_error=False,
                                                fill_value='extrapolate')

            y_estimate_interp[interfaces_key] = interpolator(sorted(x))

        # depth_estim.append(depth_interface_estimate_mean/depth_interface_estimate_count)
        depth_estim.append(depth_detected)
        depth_true.append(depth_detected_true)
        depth_estim_accuracy.append(depth_interface_accuracy)
        depth_estim_labels.append(
            test_name_pred + '_' + str(test_definitions.test_parameters[test_name_pred]['layers_pos'][0]))

        # slopestabilityML.plot_class_overview(test_results[test_name_pred], test_name_pred, class_in, y_pred, clf_name, depth_estimate=depth_interface_estimate,
        #                                     depth_accuracy=depth_interface_accuracy)

        slopestabilityML.plot_class_overview(test_results[test_name_pred], test_name_pred_orig, class_in, y_pred,
                                             clf_name, training=False, depth_estimate=depth_interface_estimate,
                                             interface_y=y_estimate_interp, interface_x=sorted(x),
                                             depth_accuracy=depth_interface_accuracy, batch_name=batch_name)

        # Evaluate result
        # accuracy_.append(len(np.where(y_pred == y_answer.to_numpy())) / len(y_answer.to_numpy()) * 100)
        accuracy_result.append(score * 100)
        accuracy_labels.append(test_name_pred_orig)

        # Evaluate

    return result_class, accuracy_labels, accuracy_result, depth_estim, depth_true, \
        depth_estim_accuracy, depth_estim_labels,
