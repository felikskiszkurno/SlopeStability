#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 17.01.2021

@author: Feliks Kiszkurno
"""

import matplotlib.pyplot as plt


def plot_results(accuracy_labels, accuracy_score):

    plt.figure()
    plt.scatter(accuracy_labels, accuracy_score)
    plt.ylabel('Test name')
    plt.xlabel('Correct points [%]')
    plt.title('SVM classification accuracy')
    plt.savefig('results/figures/SVM.eps')
    plt.savefig('results/figures/SVM.pdf')
    plt.savefig('results/figures/SVM.png')

    return
