#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19.01.2021

@author: Feliks Kiszkurno
"""

import settings
import slopestabilityML
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib import ticker
import slopestabilitytools


def run_classification(test_training, test_prediction, test_results, clf, clf_name):

    accuracy_result = []
    accuracy_labels = []

    accuracy_result_training = []
    accuracy_labels_training = []

    num_feat = []

    if settings.settings['norm'] is True:
        num_feat.append('RESN')
    else:
        num_feat.append('RES')

    if settings.settings['sen'] is True:
        num_feat.append('SEN')

    if settings.settings['depth'] is True:
        num_feat.append('Y')

    # if settings.settings['norm'] is True and settings.settings['sen'] is True:
    #     num_feat = ['RESN', 'SEN']
    # elif settings.settings['norm'] is False and settings.settings['sen'] is True:
    #     num_feat = ['RES', 'SEN']
    # elif settings.settings['norm'] is False and settings.settings['sen'] is False:
    #     num_feat = ['RES']
    # elif settings.settings['norm'] is True and settings.settings['sen'] is False:
    #     num_feat = ['RESN']

    #cat_feat = ['CLASS']
    #cat_lab = [0, 1]

    if settings.settings['norm_class'] is True:
        #cat_feat = ['CLASSN']
        cat_lab = np.linspace(0, settings.settings['norm_class_num'] - 1, settings.settings['norm_class_num'])

    elif settings.settings['norm_class'] is False:
        #cat_feat = ['CLASS']
        cat_lab = [0, 1]

    #cat_trans = OneHotEncoder(categories=[cat_lab])

    num_trans = StandardScaler()

    preprocessor = ColumnTransformer(transformers=[('num', num_trans, num_feat)])#,
                                                   #('cat', cat_trans, cat_feat)])

    clf_pipeline = make_pipeline(preprocessor, clf)

    for test_name in test_training:
        # Prepare data
        print(test_name)
        x_train, y_train = slopestabilityML.preprocess_data(test_results[test_name])
        # Train classifier
        clf_pipeline.fit(x_train, y_train)
        y_pred = clf_pipeline.predict(x_train)
        score_training1 = clf_pipeline.score(x_train, y_train)
        score_training = accuracy_score(y_train, y_pred)
        if score_training1 == score_training:
            print('MATCH!')
        else:
            print('MISMATCH!')
        accuracy_result_training.append(score_training * 100)
        accuracy_labels_training.append(test_name)

        slopestabilityML.plot_class_overview(test_results[test_name], test_name, y_train, y_pred, clf_name, training=True)

    result_class = {}

    # Predict with classifier
    for test_name_pred in test_prediction:
        # Prepare data
        x_question, y_answer = slopestabilityML.preprocess_data(test_results[test_name_pred])

        # y_pred = clf_pipeline.score(x_question, y_answer)
        y_pred = clf_pipeline.predict(x_question)
        result_class[test_name_pred] = y_pred
        # print(y_pred)
        score1 = clf_pipeline.score(x_question, y_answer)
        score = accuracy_score(y_answer, y_pred)
        if score1 == score:
            print('MATCH!')
        else:
            print('MISMATCH!')
        print('score: '+str(score))

        if settings.settings['norm_class'] is True:
            class_in = test_results[test_name_pred]['CLASSN']
        elif settings.settings['norm_class'] is False:
            class_in = test_results[test_name_pred]['CLASS']
        else:
            print('I don\'t know which class to use! Exiting...')
            exit(0)

        slopestabilityML.plot_class_overview(test_results[test_name_pred], test_name_pred, class_in, y_pred, clf_name)

        # Evaluate result
        #accuracy_.append(len(np.where(y_pred == y_answer.to_numpy())) / len(y_answer.to_numpy()) * 100)
        accuracy_result.append(score*100)
        accuracy_labels.append(test_name_pred)

    return result_class, accuracy_labels, accuracy_result, accuracy_labels_training, accuracy_result_training

