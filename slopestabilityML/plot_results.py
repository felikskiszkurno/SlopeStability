#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 17.01.2021

@author: Feliks Kiszkurno
"""

import matplotlib.pyplot as plt
from pathlib import Path


def plot_results(accuracy_labels, accuracy_score, clf_name):

    clf_name_title = clf_name.replace("_", " ")

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(accuracy_labels, accuracy_score, marker='x')
    plt.setp(ax.get_xticklabels(), rotation=90)
    plt.xlabel('Test name')
    plt.ylabel('Correct points [%]')
    plt.title(clf_name_title+' accuracy score')
    print('plot script is executed')
    fig.tight_layout()
    fig.savefig(Path('results/figures/ML/'+clf_name+'.eps'), bbox_inches="tight")
    fig.savefig(Path('results/figures/ML/'+clf_name+'.pdf'), bbox_inches="tight")
    fig.savefig(Path('results/figures/ML/'+clf_name+'.png'), bbox_inches="tight")

    return
