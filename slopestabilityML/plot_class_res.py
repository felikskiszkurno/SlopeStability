#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 31.03.2021

@author: Feliks Kiszkurno
"""

# LEGACY: NOW A PART OF plot_class_overview

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import slopestabilitytools


def plot_class_res(test_results, test_name_pred, class_in, y_pred, clf_name, *, training=False):
    # TODO: Move plotting to a function for plotting a, b and a-b
    x = test_results[test_name_pred]['X']
    y = test_results[test_name_pred]['Y']

    class_out = y_pred

    data = {'class_in': class_in, 'class_out': class_out}

    xi, yi, data_gridded = slopestabilitytools.grid_data(x, y, data)

    class_in_i = data_gridded['class_in']
    class_out_i = data_gridded['class_out']

    class_diff = np.zeros_like(class_out_i)
    class_diff[np.where(class_in_i == class_out_i)] = 1
    cb = []

    fig, _ax = plt.subplots(nrows=3, ncols=1)
    ax = _ax.flatten()

    fig.suptitle(test_name_pred + ' ' + clf_name)
    fig.subplots_adjust(hspace=0.8)

    im0 = ax[0].contourf(xi, yi, class_in_i)
    ax[0].set_title('Input classes')
    ax[0] = slopestabilitytools.set_labels(ax[0])
    cb.append(plt.colorbar(im0, ax=ax[0], label='Class'))  # , shrink=0.9)
    tick_locator = ticker.MaxNLocator(nbins=4)
    cb[0].locator = tick_locator
    cb[0].update_ticks()

    im1 = ax[1].contourf(xi, yi, class_out_i)
    ax[1].set_title('Result of classification')
    ax[1] = slopestabilitytools.set_labels(ax[1])
    cb.append(plt.colorbar(im1, ax=ax[1], label='Class'))  # , shrink=0.9)
    tick_locator = ticker.MaxNLocator(nbins=4)
    cb[1].locator = tick_locator
    cb[1].update_ticks()

    im2 = ax[2].contourf(xi, yi, class_diff)
    ax[2].set_title('Difference')
    ax[2] = slopestabilitytools.set_labels(ax[2])
    cb.append(plt.colorbar(im2, ax=ax[2], label='Is class correct?'))  # , shrink=0.9)
    tick_locator = ticker.MaxNLocator(nbins=4)
    cb[2].locator = tick_locator
    cb[2].update_ticks()

    fig.tight_layout()

    if training is True:
        slopestabilitytools.save_plot(fig, clf_name, '{}_ML_{}_class_res'.format(test_name_pred, clf_name),
                                    subfolder='ML/training')

    else:
        slopestabilitytools.save_plot(fig, clf_name, '{}_ML_{}_class_res'.format(test_name_pred, clf_name),
                                      subfolder='ML/prediction')

    return
