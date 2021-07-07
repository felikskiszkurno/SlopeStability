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
from sklearn.base import is_classifier

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

    if settings.settings['depth'] is True:
        num_feat.append('Y')

    if settings.settings['weight'] is True:
        try:
            num_feat.remove('SEN')
        except ValueError:
            print('SEN already removed')
            
    if settings.settings['sen'] is False:
        try:
            num_feat.remove('SEN')
        except ValueError:
            print('SEN already removed')
    
    if settings.settings['sen'] is True:
        num_feat.append('SEN')

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
        test_results_temp = test_results[name]
        if settings.settings['resample'] is False:

            if settings.settings['min_sen_train'] is True:
                sen = test_results_temp['SEN'].to_numpy()
                senn = (sen + abs(sen.min())) / (sen.max() + abs(sen.min()))
                test_results_temp['SENN'] = senn
                test_results_temp = test_results_temp[
                    test_results_temp['SENN'] > settings.settings['min_sen_train_val']]
            # test_results_combined = test_results_combined.reset_index(drop=True)

            if settings.settings['balance'] is True:

                test_results_bal = slopestabilityML.balance_classes(test_results_temp, test_name=name)
                test_results_combined = test_results_combined.append(test_results_bal)

            else:

                test_results_combined = test_results_combined.append(test_results_temp)

        elif settings.settings['resample'] is True:

            test_results_resampled = slopestabilitytools.resample_profile(test_results[name])
            if settings.settings['min_sen_train'] is True:
                sen = test_results_resampled['SEN'].to_numpy()
                senn = (sen + abs(sen.min())) / (sen.max() + abs(sen.min()))
                test_results_temp = test_results_resampled
                test_results_temp['SENN'] = senn
                test_results_temp = test_results_temp[
                    test_results_temp['SENN'] > settings.settings['min_sen_train_val']]

            if settings.settings['balance'] is True:

                test_results_bal = slopestabilityML.balance_classes(test_results_temp, test_name=name)
                test_results_combined = test_results_combined.append(test_results_bal)

            else:

                test_results_combined = test_results_combined.append(test_results_temp)

    test_results_combined = test_results_combined.reset_index(drop=True)

    # Apply bh simulation
    if settings.settings['sim_bh'] is True:
        test_results_combined_orig = test_results_combined.copy()
        test_results_combined = pd.DataFrame(columns=test_results_combined_orig.columns.values.tolist())
        for borehole_id in settings.settings['bh_pos'].keys():
            bh_pos = settings.settings['bh_pos'][borehole_id]
            df_temp = test_results_combined_orig.loc[(test_results_combined_orig['X'] > bh_pos['x_start']) &
                                                     (test_results_combined_orig['X'] < bh_pos['x_end'])&
                                                     (test_results_combined_orig['Y'] > bh_pos['y_start']) &
                                                     (test_results_combined_orig['Y'] < bh_pos['y_end'])]
            test_results_combined = test_results_combined.append(df_temp)
            # test_results_combined.append(test_results_combined_orig[index_bh])
            # del index_bh
    test_results_combined = test_results_combined.reset_index(drop=True)

    # test_results_combined = test_results_combined.drop(['index'], axis='columns')
    # here will balance_class be called

    # test_results_combined = test_results_combined.reset_index(drop=True)
    # Sample to reduce amount of data
    if settings.settings['reduce_samples'] is True:
        test_resampled = pd.DataFrame(columns=test_results_combined.columns.values.tolist())
        temp_df = pd.DataFrame(pd.DataFrame(columns=test_results_combined.columns.values.tolist()))
        for test_name in test_training:
            temp_df = test_results_combined.loc[test_results_combined['NAME'] == test_name]
            temp_df_resampled = temp_df.sample(frac=settings.settings['reduce_samples_factor'])
            test_resampled = test_resampled.append(temp_df_resampled)
        test_results_combined_bh = test_results_combined.copy()
        test_results_combined = test_resampled.copy()
    test_results_combined = test_results_combined.reset_index(drop=True)
    x_train, y_train, x_position = slopestabilityML.preprocess_data(test_results_combined, return_x=True)

    # x_position = test_results_combined['X']

    x_train = x_train[num_feat]
    # x_train = x_train.reset_index()
    # x_train = x_train.drop(['index'], axis='columns')

    print('Training ' + clf_name)

    weights_np = test_results_combined['SEN'].to_numpy()
    weights = (weights_np + abs(weights_np.min())) / (weights_np.max() + abs(weights_np.min()))

    if settings.settings['weight'] is True:
        # x_train = x_train.drop(['SEN'], axis='columns')
        # weights = pd.DataFrame(weights_norm)
        # weights = x_train['SEN']
        # x_train.pop('SEN')
        try:
            clf_pipeline.fit(x_train, y_train, **{clf_pipeline.steps[-1][-1].estimator + '__sample_weight': weights})
        except TypeError:
            try:
                clf_pipeline.fit(x_train, y_train)
            except TypeError:
                clf_pipeline.fit(x_train, y_train, scoring="accuracy")
        except AttributeError:
            try:
                print('No support for sample_weight... running classifier without it!')
                clf_pipeline.fit(x_train, y_train)
            except TypeError:
                print('No support for sample_weight... running classifier without it!')
                clf_pipeline.fit(x_train, y_train, scoring="accuracy")

    else:
        clf_pipeline.fit(x_train, y_train)


    clf_name_ext = clf_name + '.sav'
    clf_file_name = os.path.join(settings.settings['clf_folder'], clf_name_ext)
    joblib.dump(clf_pipeline, clf_file_name)
    settings.settings['clf_trained'] = slopestabilitytools.find_clf()

    confusion_matrix_sum = np.zeros([settings.settings['norm_class_num'],
                                     settings.settings['norm_class_num']])
    # Prediction to obtain quality of training figures and stuff
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
        slopestabilityML.plot_sen_corr(y_pred, class_correct, test_results_combined['SEN'].loc[index], clf_name, name,
                                       'training', training=True)
        if not is_classifier(clf):
            print('Skipping confusion matrix as clssifier {} doesnt support it...'.format(clf_name))
        else:
            conf_matr_temp = slopestabilityML.plot_confusion(clf_name, clf, y_pred=x_train_temp,
                                                             y_true=class_correct,
                                                             test_name=name, training=True)

            confusion_matrix_sum = confusion_matrix_sum + conf_matr_temp

        result_class_training[name] = y_pred
        score_training = accuracy_score(class_correct, y_pred)
        accuracy_result_training.append(score_training * 100)
        accuracy_labels_training.append(name)

        if not is_classifier(clf):
            print('Skipping feature importance as clssifier {} doesnt support it...'.format(clf_name))
        else:
            importance = permutation_importance(clf_pipeline, x_train_temp, y_pred)
            slopestabilityML.plot_feature_importance(clf_name, importance, x_train_temp, name)
            slopestabilityML.plot_feature_importance(clf_name, importance, x_train_temp, name)
        # log_file_name = settings.settings['log_file_name']
        # log_file = open(os.path.join(settings.settings['results_folder'], log_file_name), 'a')
        # log_file.write('\n')
        # log_file.write('Starting training on profile: {tn}'.format(tn=name))
        # log_file.write('\n')
        # log_file.write('{tn} score: {score:.2f} %'.format(tn=name, score=score_training * 100))
        # log_file.write('\n')
        # log_file.write('{tn} feature list: {fl}'.format(tn=name,
        #                                                 fl=x_train_temp.columns.values.tolist()))
        # log_file.write('\n')
        # #log_file.write('{tn}  feature importance: {fi}'.format(tn=name,
        # #                                                       fi=importance.importances_mean))
        # log_file.write('\n')
        # log_file.close()



        # Evaluate the accuracy of interface depth detection
        x_temp = x_position.loc[index].to_numpy()
        x = x_temp.reshape(len(x_temp))
        y = test_results_combined['Y'].loc[index]
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
        x_interpolator = {}
        depth_interface_estimate_count = 0
        depth_interface_accuracy_mean = 0
        depth_interface_estimate_mean = 0
        depth_interface_accuracy = 0
        depth_interface_true = test_definitions.test_parameters[name]['layers_pos']
        depth_detected_train = []
        depth_detected_true_train = []
        for interfaces_key in interfaces_detected.keys():
            diff = abs(np.ones([len(depth_interface_true)]) * interfaces_detected[interfaces_key][
                'depth_mean'] - depth_interface_true)
            # print(depth_interface_true)
            # print(interfaces_detected[interfaces_key])
            # print(diff)
            best_match_id = np.argwhere(diff == np.min(diff))
            # print(best_match_id)
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
            depth_interface_accuracy = (depth_interface_accuracy / abs(best_match_depth[0])) * 100
            depth_interface_accuracy_mean += depth_interface_accuracy
            depth_interface_estimate_count += 1

            interpolator = interpolate.interp1d(x_estimate[np.isfinite(y_estimate)],
                                                y_estimate[np.isfinite(y_estimate)])  # , fill_value='extrapolate')
            x_interpolator[interfaces_key] = sorted(
                x[(x > x_estimate[np.isfinite(y_estimate)].min()) & (x < x_estimate[np.isfinite(y_estimate)].max())])
            y_estimate_interp[interfaces_key] = interpolator(x_interpolator[interfaces_key])

        if depth_interface_estimate_count == 0:
            depth_interface_accuracy_mean = 0
        else:
            depth_interface_accuracy_mean = depth_interface_accuracy_mean / depth_interface_estimate_count
        # depth_interface_accuracy = depth_interface_accuracy_mean / depth_interface_estimate_count#
        # depth_estim_training.append(depth_interface_estimate_mean/depth_interface_estimate_count)
        depth_estim_training.append(depth_detected_train)
        depth_true_training.append(depth_detected_true_train)
        depth_estim_accuracy_training.append(depth_interface_accuracy)
        depth_estim_labels_training.append(name + '_' + str(test_definitions.test_parameters[name]['layers_pos'][0]))

        slopestabilityML.plot_class_overview(test_results_combined.loc[index], name, y_train.loc[index], y_pred,
                                             clf_name, training=True, depth_estimate=depth_interface_estimate,
                                             interface_y=y_estimate_interp, interface_x=x_interpolator,
                                             depth_accuracy=depth_interface_accuracy_mean)

    # del y_pred, y_pred_grid, y_pred_grid_deri, y, x, y_actual, xi, yi, y_estimate_interp, depth_interface_accuracy
    # del depth_interface_estimate, depth_interface_accuracy_mean, depth_interface_estimate_count, depth_interface_estimate_mean

    return result_class_training, depth_estim_training, depth_true_training, depth_estim_accuracy_training, \
           depth_estim_labels_training, accuracy_result_training, accuracy_labels_training, num_feat
