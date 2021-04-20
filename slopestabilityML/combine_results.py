#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19.01.2021

@author: Feliks Kiszkurno
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import slopestabilitytools


def combine_results(ml_results):

    print('Plotting the summary...')

    # Predictions
    fig = plt.figure()
    ax = fig.subplots(1)
    fig.suptitle('Accuracy of different ML methods: predictions')

    prediction_score_sum = 0
    prediction_score_num = 0

    for method_name in sorted(ml_results.keys()):
        plt.plot(ml_results[method_name]['labels'], ml_results[method_name]['score'], marker='x',
                 label=method_name)
        prediction_score_sum = prediction_score_sum + np.sum(np.array(ml_results[method_name]['score']))
        prediction_score_num = prediction_score_num + len(ml_results[method_name]['score'])

    prediction_score_avg = prediction_score_sum / prediction_score_num
    print('Prediction accuracy: {result:.2f}%'.format(result=prediction_score_avg))

    x_limits = ax.get_xlim()
    ax.axhline(y=prediction_score_avg, xmin=x_limits[0], xmax=x_limits[1])
    plt.xlabel('Test name')
    plt.setp(ax.get_xticklabels(), rotation=45)
    plt.ylabel('Correct points [%]')
    plt.legend(loc='lower right')

    slopestabilitytools.save_plot(fig, '', 'ML_summary_prediction', skip_fileformat=True)

    # Training
    fig = plt.figure()
    ax = fig.subplots(1)
    fig.suptitle('Accuracy of different ML methods - training')

    training_score_sum = 0
    training_score_num = 0

    for method_name in sorted(ml_results.keys()):
        plt.plot(ml_results[method_name]['labels_training'], ml_results[method_name]['score_training'], marker='x',
                 label=method_name)
        training_score_sum = training_score_sum + np.sum(np.array(ml_results[method_name]['score_training']))
        training_score_num = training_score_num + len(ml_results[method_name]['score_training'])

    training_score_avg = training_score_sum / training_score_num
    print('Training accuracy: {result:.2f}%'.format(result=training_score_avg))

    x_limits = ax.get_xlim()
    plt.axhline(y=training_score_avg, xmin=x_limits[0], xmax=x_limits[1])

    plt.xlabel('Test name')
    plt.setp(ax.get_xticklabels(), rotation=90)
    plt.ylabel('Correct points [%]')
    plt.legend(loc='lower right')

    #fig.tight_layout()
    slopestabilitytools.save_plot(fig, '', 'ML_summary_training', skip_fileformat=True)
