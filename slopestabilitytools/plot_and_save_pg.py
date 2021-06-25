#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15.01.2021

@author: Feliks Kiszkurno
"""

import matplotlib.pyplot as plt
from matplotlib import ticker
import slopestabilitytools
import pygimli as pg
import numpy as np


def plot_and_save_pg(test_name, plot_title, ert_manager, input_model, result_full, class_result):

    print('Plotting and saving overview figure... ')

    # Plot data input, inversion result and difference
    # TODO: Move plotting to a function for plotting a, b and a-b
    cb = []

    fig, _ax = plt.subplots(nrows=2, ncols=2)
    ax = _ax.flatten()

    fig.suptitle(test_name + plot_title)
    fig.subplots_adjust(hspace=0.8)

    im0 = pg.show(ert_manager.paraDomain, input_model, showMesh=True, ax=ax[0], label='Resistivity \u03A9 *m')
    #im0 = ax[0].contourf(xi, yi, inm_i, vmax=rho_max, vmin=rho_min)
    ax[0].set_title('Input model')
    ax[0] = slopestabilitytools.set_labels(ax[0])
    #cb.append(plt.colorbar(im0, ax=ax[0], label='Resistivity [om]'))#, shrink=0.9)
    tick_locator = ticker.MaxNLocator(nbins=4)
    #cb[0].locator = tick_locator
    #cb[0].update_ticks()

    limits = pg.utils.interperc(ert_manager.inv.model, trimval=10.0)
    print(limits)

    im1 = pg.show(ert_manager.paraDomain, ert_manager.inv.model, cMin=limits[0], cMax=limits[1], showMesh=True, ax=ax[1],
                  label='Resistivity \u03A9 *m')
    ax[1].set_title('Result')
    ax[1] = slopestabilitytools.set_labels(ax[1])
    #cb.append(plt.colorbar(im1, ax=ax[1], label='Resistivity [om]'))#, shrink=0.9)
    #tick_locator = ticker.MaxNLocator(nbins=4)
    #cb[1].locator = tick_locator
    #cb[1].update_ticks()

    im2 = pg.show(ert_manager.paraDomain,
                  input_model-result_full, ax=ax[2], label='Resistivity \u03A9 *m')
    ax[2].set_title('Difference')
    ax[2] = slopestabilitytools.set_labels(ax[2])
    #cb.append(plt.colorbar(im2, ax=ax[2], label='Resistivity [om]'))#, shrink=0.9)
    #tick_locator = ticker.MaxNLocator(nbins=4)
    #cb[2].locator = tick_locator
    #cb[2].update_ticks()

    im3 = pg.show(ert_manager.paraDomain, ert_manager.coverage(), ax=ax[3])
    ax[3].set_title(plot_title + ' coverage')
    ax[3] = slopestabilitytools.set_labels(ax[3])
    #cb_cov = plt.colorbar(im3, ax=ax[3], label='Coverage')  # , shrink=0.9)
    #tick_locator = ticker.MaxNLocator(nbins=4)
    #cb_cov.locator = tick_locator
    #cb_cov.update_ticks()


    #cb.append(plt.colorbar(im5, ax=ax[5], label='Sensitivity'))  # , shrink=0.9)
    #tick_locator = ticker.MaxNLocator(nbins=4)
    #cb[5].locator = tick_locator
    #cb[5].update_ticks()

    fig.tight_layout()
    slopestabilitytools.save_plot(fig, test_name, plot_title)

    # Plot coverage
    # cb_cov = []
    # fig_cov, ax_cov = plt.subplots(nrows=1, ncols=1)
    #
    # fig.suptitle(plot_title+' coverage')
    #
    # im0 = ax_cov.scatter(x, y, c=sen)
    # #im0 = ax_cov.contourf(xi, yi, cov_i)
    # ax_cov.set_title(plot_title+' coverage')
    # ax_cov = slopestabilitytools.set_labels(ax_cov)
    # cb_cov = plt.colorbar(im0, ax=ax_cov, label='Coverage')  # , shrink=0.9)
    # tick_locator = ticker.MaxNLocator(nbins=4)
    # cb_cov.locator = tick_locator
    # cb_cov.update_ticks()
    #
    # fig_cov.tight_layout()
    # slopestabilitytools.save_plot(fig_cov, test_name, '_cov')

    return
