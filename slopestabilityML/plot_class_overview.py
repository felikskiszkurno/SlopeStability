#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08.04.2021

@author: Feliks Kiszkurno
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

import slopestabilitytools


def plot_class_overview(test_results, test_name_pred, class_in, y_pred, clf_name, *, training=False):

    x = test_results['X']
    y = test_results['Y']
    data = {'class_in': class_in, 'class_out': y_pred, 'sen': test_results['sen'], 'depth': y,
            'input': test_results['INMN']}

    xi, yi, data_gridded = slopestabilitytools.grid_data(x, y, data)

    # Create plot
    fig, _ax = plt.subplots(nrows = 3, ncols=2)
    ax = _ax.flatten
    cb = []

    fig.suptitle('Classification overview: '+test_name_pred+', '+clf_name)
    fig.subplots_adjust(hspace=0.8)

    # Plot input classes
    im0 = ax[0].contourf(xi, yi, data_gridded['class_in'])
    ax[0].set_title('Input classes')
    ax[0] = slopestabilitytools.set_labels(ax[0])
    cb.append(plt.colorbar(im0, ax=ax[0], label='Class'))  # , shrink=0.9)
    tick_locator = ticker.MaxNLocator(nbins=4)
    cb[0].locator = tick_locator
    cb[0].update_ticks()

    # Plot prediction
    im0 = ax[3].contourf(xi, yi, data_gridded['class_out'])
    ax[3].set_title('Predicted classes')
    ax[3] = slopestabilitytools.set_labels(ax[3])
    cb.append(plt.colorbar(im0, ax=ax[3], label='Class'))  # , shrink=0.9)
    tick_locator = ticker.MaxNLocator(nbins=4)
    cb[3].locator = tick_locator
    cb[3].update_ticks()

    # Plot input model
    im0 = ax[1].contourf(xi, yi, data_gridded['INMN'])
    ax[1].set_title('Input model')
    ax[1] = slopestabilitytools.set_labels(ax[1])
    cb.append(plt.colorbar(im0, ax=ax[1], label='Resistivity log(ohm*m)'))  # , shrink=0.9)
    tick_locator = ticker.MaxNLocator(nbins=4)
    cb[1].locator = tick_locator
    cb[1].update_ticks()

    # Plot sensitivity
    im0 = ax[4].contourf(xi, yi, data_gridded['sen'])
    ax[4].set_title('Input classes')
    ax[4] = slopestabilitytools.set_labels(ax[4])
    cb.append(plt.colorbar(im0, ax=ax[4], label='Sensitivity'))  # , shrink=0.9)
    tick_locator = ticker.MaxNLocator(nbins=4)
    cb[4].locator = tick_locator
    cb[4].update_ticks()

    # Plot depth
    im0 = ax[5].contourf(xi, yi, data_gridded['depth'])
    ax[5].set_title('Input classes')
    ax[5] = slopestabilitytools.set_labels(ax[5])
    cb.append(plt.colorbar(im0, ax=ax[5], label='Depth [m]'))  # , shrink=0.9)
    tick_locator = ticker.MaxNLocator(nbins=4)
    cb[5].locator = tick_locator
    cb[5].update_ticks()

    fig.tight_layout()

    if training is True:
        slopestabilitytools.save_plot(fig, clf_name, '{}_ML_{}_class_overview'.format(clf_name, test_name_pred),
                                    subfolder='ML/training')

    else:
        slopestabilitytools.save_plot(fig, clf_name, '{}_ML_{}_class_overview'.format(clf_name, test_name_pred),
                                      subfolder='ML/prediction')

    return
