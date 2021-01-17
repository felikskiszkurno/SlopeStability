#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15.01.2021

@author: Feliks Kiszkurno
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import slopestabilitytools


def plot_and_save(test_name, test_result, plot_title):

    x = test_result['X']
    print(x)
    y = test_result['Y']
    inm = test_result['INM']
    res = test_result['RES']

    x_vec = np.unique(np.array(x))
    y_vec = np.unique(np.array(y))
    print(y_vec)
    X, Y = np.meshgrid(x_vec, y_vec)
    [m, n] = X.shape
    inm_plot = np.array(inm).reshape((m, n))
    res_plot = np.array(res).reshape((m, n))
    print('plot_and_save')
    print(X)

    fig, ax = plt.subplots(3)

    fig.suptitle(plot_title)

    ax[0].contourf(X, Y, inm_plot)
    ax[0].set_title('Input model')
    ax[0] = slopestabilitytools.set_labels(ax[0])

    ax[1].contourf(X, Y, res_plot)
    ax[1].set_title('Result')
    ax[1] = slopestabilitytools.set_labels(ax[0])

    ax[2].contourf(X, Y, inm_plot-res_plot)
    ax[2].set_title('Difference')
    ax[2] = slopestabilitytools.set_labels(ax[0])

    fig.savefig('results/figures/eps/hor_{}_input.eps'.format(test_name))
    fig.savefig('results/figures/png/hor_{}_input.png'.format(test_name))
    fig.savefig('results/figures/pdf/hor_{}_input.pdf'.format(test_name))

    return
