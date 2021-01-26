#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19.01.2021

@author: Feliks Kiszkurno
"""

import slopestabilityML
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy import interpolate
import slopestabilitytools
from pathlib import Path


def run_classification(test_training, test_prediction, test_results, clf, clf_name):

    accuracy_score = []
    accuracy_labels = []

    accuracy_score_training = []
    accuracy_labels_training = []

    num_feat = ['RES', 'SEN']
    #cat_feat = ['CLASS']

    cat_lab = [0, 1]

    num_trans = StandardScaler()
    #cat_trans = OneHotEncoder(categories=[cat_lab])

    preprocessor = ColumnTransformer(transformers=[('num', num_trans, num_feat),])
                                                   #('cat', cat_trans, cat_feat)])

    clf_pipeline_UM = make_pipeline(preprocessor, clf)

    for test_name in test_training:
        # Prepare data
        x_train, y_train = slopestabilityML.preprocess_data(test_results[test_name])

        # Train classifier
        # print(type(x_train))
        # print(type(y_train))
        clf_pipeline_UM.fit(x_train, y_train)
        score_training = clf_pipeline_UM.score(x_train, y_train)

        accuracy_score_training.append(score_training * 100)
        accuracy_labels_training.append(test_name)

    # Predict with classifier
    for test_name_pred in test_prediction:
        # Prepare data
        x_question, y_answer = slopestabilityML.preprocess_data(test_results[test_name_pred])

        # y_pred = clf_pipeline_UM.score(x_question, y_answer)
        y_pred = clf_pipeline_UM.predict(x_question)
        # print(y_pred)
        score = clf_pipeline_UM.score(x_question, y_answer)
        print('score: '+str(score))

        # TODO: Move plotting to a function for plotting a, b and a-b
        x = test_results[test_name_pred]['X']
        y = test_results[test_name_pred]['Y']
        class_in = test_results[test_name]['CLASS']
        class_out = y_pred
        x_min = np.min(x)
        x_max = np.max(x)
        x_n = len(x)

        y_min = np.min(y)
        y_max = np.max(y)
        y_start = y_max
        y_end = y_min
        y_n = len(y)

        xi = np.linspace(x_min, x_max, x_n)
        yi = np.linspace(y_start, y_end, y_n)
        xx, yy = np.meshgrid(xi, yi)
        class_in_i = interpolate.griddata((x, y), class_in, (xx, yy), method='nearest')
        class_out_i = interpolate.griddata((x, y), class_out, (xx, yy), method='nearest')
        class_diff = np.zeros_like(class_out_i)
        class_diff[np.where(class_in_i == class_out_i)] = 1
        cb = []

        fig, _ax = plt.subplots(nrows=3, ncols=1)
        ax = _ax.flatten()

        fig.suptitle(test_name_pred+' '+clf_name)
        fig.subplots_adjust(hspace=0.8)

        im0 = ax[0].contourf(xi, yi, class_in_i)
        ax[0].set_title('Input classes')
        ax[0] = slopestabilitytools.set_labels(ax[0])
        cb.append(plt.colorbar(im0, ax=ax[0], label='Class'))  # , shrink=0.9)
        tick_locator = ticker.MaxNLocator(nbins=4)
        cb[0].locator = tick_locator
        cb[0].update_ticks()

        im1 = ax[1].contourf(xi, yi, class_out_i)
        ax[1].set_title('Result of classification')
        ax[1] = slopestabilitytools.set_labels(ax[1])
        cb.append(plt.colorbar(im1, ax=ax[1], label='Class'))  # , shrink=0.9)
        tick_locator = ticker.MaxNLocator(nbins=4)
        cb[1].locator = tick_locator
        cb[1].update_ticks()

        im2 = ax[2].contourf(xi, yi, class_diff)
        ax[2].set_title('Difference')
        ax[2] = slopestabilitytools.set_labels(ax[2])
        cb.append(plt.colorbar(im2, ax=ax[2], label='Resistivity [om]'))  # , shrink=0.9)
        tick_locator = ticker.MaxNLocator(nbins=4)
        cb[2].locator = tick_locator
        cb[2].update_ticks()

        fig.savefig(Path('results/figures/ML/prediction/eps/{}_ML_{}_class_res.eps'.format(test_name_pred, clf_name)))
        fig.savefig(Path('results/figures/ML/prediction/png/{}_ML_{}_class_res.png'.format(test_name_pred, clf_name)))
        fig.savefig(Path('results/figures/ML/prediction/pdf/{}_ML_{}_class_res.pdf'.format(test_name_pred, clf_name)))

        # Evaluate result
        #accuracy_score.append(len(np.where(y_pred == y_answer.to_numpy())) / len(y_answer.to_numpy()) * 100)
        accuracy_score.append(score*100)
        accuracy_labels.append(test_name_pred)

    return accuracy_labels, accuracy_score, accuracy_labels_training, accuracy_score_training

