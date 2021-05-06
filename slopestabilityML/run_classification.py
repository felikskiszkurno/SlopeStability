#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19.01.2021

@author: Feliks Kiszkurno
"""
import settings
import slopestabilityML
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
import slopestabilitytools
import test_definitions


def run_classification(test_training, test_prediction, test_results, clf, clf_name):
    accuracy_result = []
    accuracy_labels = []

    accuracy_result_training = []
    accuracy_labels_training = []

    depth_estim = []
    depth_estim_accuracy = []
    depth_estim_labels = []

    depth_estim_training = []
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
        test_results_combined = test_results_combined.append(test_results[name])
    test_results_combined = test_results_combined.reset_index()
    test_results_combined = test_results_combined.drop(['index'], axis='columns')
    x_train, y_train = slopestabilityML.preprocess_data(test_results_combined)
    x_position = test_results_combined['X']

    clf_pipeline.fit(x_train, y_train)

    for name in test_training:
        print(name)
        index = test_results_combined.index[test_results_combined['NAME'] == name]
        if settings.settings['norm_class'] is True:
            class_correct = test_results_combined['CLASSN'].loc[index]
        else:
            class_correct = test_results_combined['CLASS'].loc[index]
        x_train_temp = x_train.loc[index]
        y_pred = clf_pipeline.predict(x_train_temp)
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
        # interface_id = []
        # interface_n = []
        # interface_depth = []
        # for column in y_pred_grid_deri.T:
        #     indexes = list(np.where(column != 0))
        #     if indexes[0].size == 1:  # Only one interface detected
        #         print('a')
        #         interface_n.append(len(indexes[0]))
        #         interface_id.append(indexes)
        #         interface_depth.append(yi[indexes])
        #     else:  # Multiple potential interfaces detected
        #         print('b')
        #         indexes_temp = []
        #         depths_temp = []
        #         indexes_copy = indexes
        #         all_too_thin_layers_removed = False
        #         while all_too_thin_layers_removed is False:
        #             all_too_thin_layers_removed = True
        #             if isinstance(indexes_copy[0], int) is False:
        #                 for index_id in range(indexes_copy[0].size-1):
        #                     diff = yi[indexes[0][index_id+1]] - yi[indexes[0][index_id]]
        #                     if abs(diff) <= 1:
        #                         indexes_temp.append(int(round((indexes[0][index_id+1]+indexes[0][index_id])/2)))
        #                         all_too_thin_layers_removed = False
        #                     else:
        #                         indexes_temp.append(indexes[0][index_id])
        #                 if abs(yi[indexes[0][-1]] - yi[indexes[0][-2]]) >= 1:
        #                     indexes_temp.append(indexes[0][-1])
        #                 indexes_copy = indexes_temp
        #                 indexes_temp = []
        #         interface_n.append(len(indexes_copy))
        #         interface_id.append(indexes_copy)
        #         interface_depth.append(yi[indexes_copy])
        # interface_number = np.bincount(interface_n).argmax()
        for column_id in range(y_pred_grid.shape[0]):
            # if len(np.unique(y_pred_grid[:,column_id])) is not 2:
            depth_id = np.array(np.where(y_pred_grid[:, column_id] == 4))
            if np.size(depth_id) is 0:
                depth = yi[-1]
            else:
                depth_id = depth_id.min()
                depth = yi[depth_id]
            depth_all[column_id] = depth
        depth_interface_estimate = np.mean(depth_all)
        depth_interface_accuracy = (mean_absolute_error(depth_all_correct, depth_all) / abs(test_definitions.test_parameters[name]['layers_pos'][0]))*100
        print(depth_interface_accuracy)
        depth_estim_training.append(depth_interface_estimate)
        depth_estim_accuracy_training.append(depth_interface_accuracy)
        depth_estim_labels_training.append(name + '_' + str(test_definitions.test_parameters[name]['layers_pos'][0]))
        # print(y_train.loc[index])
        slopestabilityML.plot_class_overview(test_results_combined.loc[index], name, y_train.loc[index], y_pred,
                                             clf_name, training=True, depth_estimate=depth_interface_estimate,
                                             depth_accuracy=depth_interface_accuracy)

    result_class = {}

    # Predict with classifier
    for test_name_pred in test_prediction:
        # Prepare data
        print(test_name_pred)
        x_question, y_answer = slopestabilityML.preprocess_data(test_results[test_name_pred])
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
        x = x_position.loc[index].to_numpy()
        y = x_train_temp['Y'].to_numpy()
        xi, yi, gridded_data = slopestabilitytools.grid_data(x, y, {'class': y_pred})
        y_pred_grid = gridded_data['class']
        depth_all = np.zeros(y_pred_grid.shape[0])
        depth_all_correct = np.ones(y_pred_grid.shape[0]) * test_definitions.test_parameters[test_name_pred]['layers_pos'][0]
        for column_id in range(y_pred_grid.shape[0]):
            # if len(np.unique(y_pred_grid[:,column_id])) is not 2:
            depth_id = np.array(np.where(y_pred_grid[:, column_id] == 4))
            if np.size(depth_id) is 0:
                depth = yi[-1]
            else:
                depth_id = depth_id.min()
                depth = yi[depth_id]
            depth_all[column_id] = depth
        depth_interface_estimate = np.mean(depth_all)

        depth_interface_accuracy = (mean_absolute_error(depth_all_correct, depth_all) / abs(test_definitions.test_parameters[name]['layers_pos'][0]))*100
        print(depth_interface_accuracy)
        depth_estim.append(depth_interface_estimate)
        depth_estim_accuracy.append(depth_interface_accuracy)
        depth_estim_labels.append(test_name_pred + '_' + str(test_definitions.test_parameters[test_name_pred]['layers_pos'][0]))

        slopestabilityML.plot_class_overview(test_results[test_name_pred], test_name_pred, class_in, y_pred, clf_name, depth_estimate=depth_interface_estimate,
                                             depth_accuracy=depth_interface_accuracy)

        # Evaluate result
        # accuracy_.append(len(np.where(y_pred == y_answer.to_numpy())) / len(y_answer.to_numpy()) * 100)
        accuracy_result.append(score * 100)
        accuracy_labels.append(test_name_pred)

        # Evaluate

    return result_class, accuracy_labels, accuracy_result, accuracy_labels_training, accuracy_result_training, depth_estim, depth_estim_accuracy, depth_estim_labels, depth_estim_training, depth_estim_accuracy_training, depth_estim_labels_training
