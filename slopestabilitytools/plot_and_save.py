#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15.01.2021

@author: Feliks Kiszkurno
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy import interpolate
import numpy as np
import slopestabilitytools


def plot_and_save(test_name, test_result, plot_title):

    x = test_result['X']
    y = test_result['Y']
    inm = test_result['INM']
    res = test_result['RES']
    cov = test_result['SEN']

    xx, yy = np.meshgrid(x, y)
    inm_i = interpolate.griddata((x, y), inm, (xx, yy), method='linear')
    res_i = interpolate.griddata((x, y), res, (xx, yy), method='linear')
    cov_i = interpolate.griddata((x, y), cov, (xx, yy), method='linear')

    print('plot_and_save')

    # Plot data input, inversion result and difference
    cb = []

    fig, _ax = plt.subplots(nrows=3, ncols=1)
    ax = _ax.flatten()

    fig.suptitle(plot_title)
    fig.subplots_adjust(hspace=0.8)

    im0 = ax[0].contourf(xx, yy, inm_i)
    ax[0].set_title('Input model')
    ax[0] = slopestabilitytools.set_labels(ax[0])
    cb.append(plt.colorbar(im0, ax=ax[0], label='Resistivity [om]'))#, shrink=0.9)
    tick_locator = ticker.MaxNLocator(nbins=4)
    cb[0].locator = tick_locator
    cb[0].update_ticks()

    im1 = ax[1].contourf(xx, yy, res_i)
    ax[1].set_title('Result')
    ax[1] = slopestabilitytools.set_labels(ax[1])
    cb.append(plt.colorbar(im1, ax=ax[1], label='Resistivity [om]'))#, shrink=0.9)
    tick_locator = ticker.MaxNLocator(nbins=4)
    cb[1].locator = tick_locator
    cb[1].update_ticks()

    im2 = ax[2].contourf(xx, yy, inm_i-res_i)
    ax[2].set_title('Difference')
    ax[2] = slopestabilitytools.set_labels(ax[2])
    cb.append(plt.colorbar(im2, ax=ax[2], label='Resistivity [om]'))#, shrink=0.9)
    tick_locator = ticker.MaxNLocator(nbins=4)
    cb[2].locator = tick_locator
    cb[2].update_ticks()

    fig.savefig('results/figures/eps/{}_in_inv_diff.eps'.format(test_name))
    fig.savefig('results/figures/png/{}_in_inv_diff.png'.format(test_name))
    fig.savefig('results/figures/pdf/{}_in_inv_diff.pdf'.format(test_name))

    # Plot coverage
    cb_cov = []
    fig_cov, ax_cov = plt.subplots(nrows=1, ncols=1)

    fig.suptitle(plot_title+' coverage')

    im0 = ax_cov.contourf(xx, yy, cov_i)
    ax_cov.set_title(plot_title+' coverage')
    ax_cov = slopestabilitytools.set_labels(ax_cov)
    cb_cov = plt.colorbar(im0, ax=ax_cov, label='Coverage')  # , shrink=0.9)
    tick_locator = ticker.MaxNLocator(nbins=4)
    cb_cov.locator = tick_locator
    cb_cov.update_ticks()

    fig_cov.savefig('results/figures/eps/{}_cov.eps'.format(test_name))
    fig_cov.savefig('results/figures/png/{}_cov.png'.format(test_name))
    fig_cov.savefig('results/figures/pdf/{}_cov.pdf'.format(test_name))

    return
