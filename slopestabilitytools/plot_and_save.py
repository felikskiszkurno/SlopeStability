#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15.01.2021

@author: Feliks Kiszkurno
"""

import pandas as pd
import matplotlib.pyplot as plt
import slopestabilitytools


def plot_and_save(test_name, test_result, plot_title):

    x = test_result['X']
    print(x)
    y = test_result['Y']
    inm = test_result['INM']
    res = test_result['RES']

    fig, ax = plt.subplots(3)

    fig.suptitle(plot_title)
    
    ax[0].scatter(x, y, inm)
    ax[0].set_title('Input model')
    ax[0] = slopestabilitytools.set_labels(ax[0])

    ax[1].scatter(x, y, res)
    ax[1].set_title('Result')
    ax[1] = slopestabilitytools.set_labels(ax[0])

    ax[2].scatter(x, y, inm-res)
    ax[2].set_title('Difference')
    ax[2] = slopestabilitytools.set_labels(ax[0])

    fig.savefig('results/figures/eps/hor_{}_input.eps'.format(test_name))
    fig.savefig('results/figures/png/hor_{}_input.png'.format(test_name))
    fig.savefig('results/figures/pdf/hor_{}_input.pdf'.format(test_name))

    return
