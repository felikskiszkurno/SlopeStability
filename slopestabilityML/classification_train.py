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


def classification_train(test_training, test_results, clf, clf_name):

    result_class_training = {}

    accuracy_result_training = []
    accuracy_labels_training = []

    depth_estim_training = []
    depth_true_training = []
    depth_estim_accuracy_training = []
    depth_estim_labels_training = []

    num_feat = []

    if settings.settings['norm'] is True:
        num_feat.append('RESN')
    else:
        num_feat.append('RES')

    if settings.settings['sen'] is True:
        num_feat.append('SEN')

    if settings.settings['depth'] is True:
        num_feat.append('Y')

    num_trans = StandardScaler()

    if settings.settings['use_labels'] is True:
        cat_feat = ['LABELS']
        cat_lab = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        cat_trans = OneHotEncoder(categories=[cat_lab])
        preprocessor = ColumnTransformer(transformers=[('num', num_trans, num_feat)])
        # ('cat', cat_trans, cat_feat)])

    else:
        preprocessor = ColumnTransformer(transformers=[('num', num_trans, num_feat)])

    clf_pipeline = make_pipeline(preprocessor, clf)

    test_results_combined = pd.DataFrame()
    for name in test_training:
        print('Training on: ' + name)
        if settings.settings['resample'] is False:
            test_results_combined = test_results_combined.append(test_results[name])
        elif settings.settings['resample'] is True:
            test_result_resampled = slopestabilitytools.resample_profile(test_results[name])
            test_results_combined = test_results_combined.append(test_result_resampled)
    test_results_combined = test_results_combined.reset_index()

    if settings.settings['sim_bh'] is True:
        test_results_combined_orig = test_results_combined.copy()
        test_results_combined = pd.DataFrame(columns=test_results_combined_orig.columns.values.tolist())
        for borehole_id in settings.settings['bh_pos'].keys():
            bh_pos = settings.settings['bh_pos'][borehole_id]
            df_temp = test_results_combined_orig.loc[(test_results_combined_orig['X'] > bh_pos['x_start']) &
                                                     (test_results_combined_orig['X'] < bh_pos['x_end']) &
                                                     (test_results_combined_orig['Y'] > bh_pos['y_start']) &
                                                     (test_results_combined_orig['Y'] < bh_pos['y_end'])]
            test_results_combined = test_results_combined.append(df_temp)
            # test_results_combined.append(test_results_combined_orig[index_bh])
            # del index_bh

    test_results_combined = test_results_combined.drop(['index'], axis='columns')
    x_train, y_train = slopestabilityML.preprocess_data(test_results_combined)
    x_position = test_results_combined['X']

    clf_pipeline.fit(x_train, y_train)

    clf_name_ext = clf_name + '.sav'
    clf_file_name = os.path.join(settings.settings['clf_folder'], clf_name_ext)
    joblib.dump(clf_pipeline, clf_file_name)
    settings.settings['clf_trained'] = slopestabilitytools.find_clf()

    for name in test_training:
        print(name)
        name_orig = name
        name, name_orig = slopestabilityML.check_name(name)
        index = test_results_combined.index[test_results_combined['NAME'] == name]
        if settings.settings['norm_class'] is True:
            class_correct = test_results_combined['CLASSN'].loc[index]
        else:
            class_correct = test_results_combined['CLASS'].loc[index]
        x_train_temp = x_train.loc[index]
        y_pred = clf_pipeline.predict(x_train_temp)
        result_class_training[name] = y_pred
        score_training = accuracy_score(class_correct, y_pred)
        accuracy_result_training.append(score_training * 100)
        accuracy_labels_training.append(name)

        # Evaluate the accuracy of interface depth detection
        x = x_position.loc[index].to_numpy()
        y = x_train_temp['Y'].to_numpy()
        xi, yi, gridded_data = slopestabilitytools.grid_data(x, y, {'class': y_pred})
        y_pred_grid = gridded_data['class']
        depth_all = np.zeros(y_pred_grid.shape[0])
        depth_all_correct = np.ones(y_pred_grid.shape[0]) * test_definitions.test_parameters[name]['layers_pos'][0]
        result_grid_rolled = np.roll(y_pred_grid, -1, axis=0)
        y_pred_grid_deri = y_pred_grid - result_grid_rolled
        y_pred_grid_deri[-1, :] = 0
        interfaces_detected = slopestabilitytools.detect_interface(xi, yi, y_pred_grid)
        depth_interface_estimate = {}
        y_estimate_interp = {}
        depth_interface_estimate_count = 0
        depth_interface_accuracy_mean = 0
        depth_interface_estimate_mean = 0
        depth_interface_true = test_definitions.test_parameters[name]['layers_pos']
        depth_detected_train = []
        depth_detected_true_train = []
        for interfaces_key in interfaces_detected.keys():
            diff = abs(np.ones([len(depth_interface_true)]) * interfaces_detected[interfaces_key][
                'depth_mean'] - depth_interface_true)
            best_match_id = np.argwhere(diff == np.min(diff))
            best_match_depth = depth_interface_true[best_match_id][0]
            depth_interface_estimate[interfaces_key] = interfaces_detected[interfaces_key]['depth_mean']
            depth_detected_train.append(interfaces_detected[interfaces_key]['depth_mean'])
            depth_detected_true_train.append(best_match_depth[0])
            depth_interface_estimate_mean = depth_interface_estimate_mean + interfaces_detected[0]['depth_mean']
            y_estimate = interfaces_detected[interfaces_key]['depths']
            x_estimate = interfaces_detected[interfaces_key]['x']
            # depth_interface_accuracy = ((depth_interface_estimate-test_definitions.test_parameters[name]['layers_pos'][0])/test_definitions.test_parameters[name]['layers_pos'][0])*100
            y_actual = np.ones([y_estimate.size]) * \
                       best_match_depth
            y_actual = y_actual.reshape([y_actual.shape[0]])
            depth_interface_accuracy = mean_squared_error(y_actual[np.isfinite(y_estimate)],
                                                          y_estimate[np.isfinite(y_estimate)],
                                                          squared=False)
            depth_interface_accuracy_mean += depth_interface_accuracy
            depth_interface_estimate_count += 1
            interpolator = interpolate.interp1d(x_estimate[np.isfinite(y_estimate)],
                                                y_estimate[np.isfinite(y_estimate)],
                                                bounds_error=False)  # , fill_value='extrapolate')
            y_estimate_interp[interfaces_key] = interpolator(sorted(x))

        # depth_estim_training.append(depth_interface_estimate_mean/depth_interface_estimate_count)
        depth_estim_training.append(depth_detected_train)
        depth_true_training.append(depth_detected_true_train)
        depth_estim_accuracy_training.append(depth_interface_accuracy)
        depth_estim_labels_training.append(name + '_' + str(test_definitions.test_parameters[name]['layers_pos'][0]))

        slopestabilityML.plot_class_overview(test_results_combined.loc[index], name, y_train.loc[index], y_pred,
                                             clf_name, training=True, depth_estimate=depth_interface_estimate,
                                             interface_y=y_estimate_interp, interface_x=x,
                                             depth_accuracy=depth_interface_accuracy)

    del y_pred, y_pred_grid, y_pred_grid_deri, y, x, y_actual, xi, yi, y_estimate_interp, depth_interface_accuracy
    del depth_interface_estimate, depth_interface_accuracy_mean, depth_interface_estimate_count, depth_interface_estimate_mean

    return result_class_training, depth_estim_training, depth_true_training, depth_estim_accuracy_training,\
        depth_estim_labels_training, accuracy_result_training, accuracy_labels_training