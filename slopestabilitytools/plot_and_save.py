#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15.01.2021

@author: Feliks Kiszkurno
"""

import matplotlib.pyplot as plt
from matplotlib import ticker
import slopestabilitytools
import settings


def plot_and_save(test_name, test_result, plot_title, rho_max, rho_min):

    x = test_result['X']
    y = test_result['Y']

    if settings.settings['norm'] is True:
        plot_title = plot_title + '_norm'
        inm = test_result['INMN']
        res = test_result['RESN']
        sen = test_result['SEN']
    elif settings.settings['norm'] is False:
        inm = test_result['INM']
        res = test_result['RES']
        sen = test_result['SEN']
    # if settings.settings['norm'] is True:
    #     plot_title = plot_title + '_norm'
    #     data = {'INM': test_result['INMN'],
    #             'RES': test_result['RESN'],
    #             'SEN': test_result['SEN']}
    # elif settings.settings['norm'] is False:
    #     data = {'INM': test_result['INM'],
    #             'RES': test_result['RES'],
    #             'SEN': test_result['SEN']}
    #
    # else:
    #     print('I don\'t know which kind of data to use! Exiting...')
    #     exit(0)
    #
    # xi, yi, data_gridded = slopestabilitytools.grid_data(x, y, data)
    #
    # inm_i = data_gridded['INM']
    # res_i = data_gridded['RES']
    # cov_i = data_gridded['SEN']

    print('Plotting and saving overview figure... ')

    # Plot data input, inversion result and difference
    # TODO: Move plotting to a function for plotting a, b and a-b
    cb = []

    fig, _ax = plt.subplots(nrows=3, ncols=1)
    ax = _ax.flatten()

    fig.suptitle(plot_title)
    fig.subplots_adjust(hspace=0.8)

    im0 = ax[0].scatter(x, y, c=inm)
    #im0 = ax[0].contourf(xi, yi, inm_i, vmax=rho_max, vmin=rho_min)
    ax[0].set_title('Input model')
    ax[0] = slopestabilitytools.set_labels(ax[0])
    cb.append(plt.colorbar(im0, ax=ax[0], label='Resistivity [om]'))#, shrink=0.9)
    tick_locator = ticker.MaxNLocator(nbins=4)
    cb[0].locator = tick_locator
    cb[0].update_ticks()

    im1 = ax[1].scatter(x, y, c=res)
    #im1 = ax[1].contourf(xi, yi, res_i, vmax=rho_max, vmin=rho_min)
    ax[1].set_title('Result')
    ax[1] = slopestabilitytools.set_labels(ax[1])
    cb.append(plt.colorbar(im1, ax=ax[1], label='Resistivity [om]'))#, shrink=0.9)
    tick_locator = ticker.MaxNLocator(nbins=4)
    cb[1].locator = tick_locator
    cb[1].update_ticks()

    im2 = ax[2].scatter(x, y, c=inm-res, cmap='RdBu')
    #im2 = ax[2].contourf(xi, yi, inm_i-res_i, cmap='RdBu')
    ax[2].set_title('Difference')
    ax[2] = slopestabilitytools.set_labels(ax[2])
    cb.append(plt.colorbar(im2, ax=ax[2], label='Resistivity [om]'))#, shrink=0.9)
    tick_locator = ticker.MaxNLocator(nbins=4)
    cb[2].locator = tick_locator
    cb[2].update_ticks()

    fig.tight_layout()
    slopestabilitytools.save_plot(fig, test_name, '_in_inv_diff')

    # Plot coverage
    cb_cov = []
    fig_cov, ax_cov = plt.subplots(nrows=1, ncols=1)

    fig.suptitle(plot_title+' coverage')

    im0 = ax_cov.scatter(x, y, c=sen)
    #im0 = ax_cov.contourf(xi, yi, cov_i)
    ax_cov.set_title(plot_title+' coverage')
    ax_cov = slopestabilitytools.set_labels(ax_cov)
    cb_cov = plt.colorbar(im0, ax=ax_cov, label='Coverage')  # , shrink=0.9)
    tick_locator = ticker.MaxNLocator(nbins=4)
    cb_cov.locator = tick_locator
    cb_cov.update_ticks()

    fig_cov.tight_layout()
    slopestabilitytools.save_plot(fig_cov, test_name, '_cov')

    return
