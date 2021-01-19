#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 17.01.2021

@author: Feliks Kiszkurno
"""

import matplotlib.pyplot as plt


def plot_results(accuracy_labels, accuracy_score, clf_name):

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.scatter(accuracy_labels, accuracy_score)
    plt.xlabel('Test name')
    plt.ylabel('Correct points [%]')
    plt.title(clf_name+' classification accuracy')
    print('plot script is executed')
    fig.savefig('results/figures/'+clf_name+'.eps')
    fig.savefig('results/figures/'+clf_name+'.pdf')
    fig.savefig('results/figures/'+clf_name+'.png')

    return
