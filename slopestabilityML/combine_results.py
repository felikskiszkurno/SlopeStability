#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19.01.2021

@author: Feliks Kiszkurno
"""

import matplotlib.pyplot as plt
from pathlib import Path


def combine_results(ml_results):

    # TODO avoid reusing the same code twice
    # Predictions
    fig = plt.figure()
    ax = fig.subplots(1)
    fig.suptitle('Accuracy of different ML methods: predictions')

    for method_name in sorted(ml_results.keys()):
        plt.plot(ml_results[method_name]['labels'], ml_results[method_name]['score'], marker='x',
                 label=method_name)

    plt.xlabel('Test name')
    plt.setp(ax.get_xticklabels(), rotation=45)
    plt.ylabel('Correct points [%]')
    plt.legend(loc='lower right')

    fig.savefig(Path('results/figures/ML_summary_prediction.eps'))
    fig.savefig(Path('results/figures/ML_summary_prediction.png'))
    fig.savefig(Path('results/figures/ML_summary_prediction.pdf'))

    # Training
    fig = plt.figure()
    ax = fig.subplots(1)
    fig.suptitle('Accuracy of different ML methods - training')

    for method_name in sorted(ml_results.keys()):
        plt.plot(ml_results[method_name]['labels_training'], ml_results[method_name]['score_training'], marker='x',
                 label=method_name)

    plt.xlabel('Test name')
    plt.setp(ax.get_xticklabels(), rotation=45)
    plt.ylabel('Correct points [%]')
    plt.legend(loc='lower right')

    fig.savefig(Path('results/figures/ML_summary_training.eps'))
    fig.savefig(Path('results/figures/ML_summary_training.png'))
    fig.savefig(Path('results/figures/ML_summary_training.pdf'))
