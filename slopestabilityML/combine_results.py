#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19.01.2021

@author: Feliks Kiszkurno
"""

import matplotlib.pyplot as plt


def combine_results(ml_results):

    fig = plt.figure()
    #ax = fig.subplots(1)
    fig.suptitle('Accuracy of different ML methods')

    for method_name in ml_results.keys():
        plt.scatter(ml_results[method_name]['labels'], ml_results[method_name]['score'], label=method_name)

    plt.legend(loc='best')

    fig.savefig('results/figures/ML_summary.eps')
    fig.savefig('results/figures/ML_summary.png')
    fig.savefig('results/figures/ML_summary.pdf')