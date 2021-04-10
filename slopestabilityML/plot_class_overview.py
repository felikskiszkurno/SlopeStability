#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08.04.2021

@author: Feliks Kiszkurno
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import gc

import slopestabilitytools
import test_definitions

def plot_class_overview(test_results, test_name, class_in, y_pred, clf_name, *, training=False):

    x = test_results['X']
    y = test_results['Y']

    class_in = class_in.to_numpy()
    class_in = class_in.reshape(class_in.size)
    data = {'class_in': class_in, 'class_out': y_pred, 'sen': test_results['SEN'].to_numpy(), 'depth': y.to_numpy(),
            'input': test_results['INMN'].to_numpy()}

    xi, yi, data_gridded = slopestabilitytools.grid_data(x, y, data)

    class_in_i = data_gridded['class_in']
    class_out_i = data_gridded['class_out']

    class_diff = np.zeros_like(class_out_i)
    class_diff[np.where(class_in_i == class_out_i)] = 1

    del data, x, y, class_in_i, class_out_i
    gc.collect()

    # Create plot
    fig, _ax = plt.subplots(nrows=4, ncols=2, figsize=(1.35*10, 10))
    ax = _ax.flatten()
    cb = []

    fig.suptitle('Classification overview: ' + test_name + ', ' + clf_name)
    fig.subplots_adjust(hspace=0.8)

    # Plot input classes
    im0 = ax[0].contourf(xi, yi, data_gridded['class_in'].reshape((data_gridded['class_in'].shape[0],
                                                                   data_gridded['class_in'].shape[1])))
    for depth in test_definitions.test_definitions[test_name]['layers_pos']:
        ax[0].hlines(y=depth, xmin=xi.min(), xmax=xi.max(), linestyle='-', color='r')
    ax[0].set_title('Input classes')
    ax[0] = slopestabilitytools.set_labels(ax[0])
    cb.append(plt.colorbar(im0, ax=ax[0], label='Class'))  # , shrink=0.9)
    tick_locator = ticker.MaxNLocator(nbins=4)
    cb[0].locator = tick_locator
    cb[0].update_ticks()

    # Plot prediction
    im1 = ax[1].contourf(xi, yi, data_gridded['class_out'])
    for depth in test_definitions.test_definitions[test_name]['layers_pos']:
        ax[1].hlines(y=depth, xmin=xi.min(), xmax=xi.max(), linestyle='-', color='r')
    ax[1].set_title('Predicted classes')
    ax[1] = slopestabilitytools.set_labels(ax[1])
    cb.append(plt.colorbar(im1, ax=ax[1], label='Class'))  # , shrink=0.9)
    tick_locator = ticker.MaxNLocator(nbins=4)
    cb[1].locator = tick_locator
    cb[1].update_ticks()

    # Plot input model
    im2 = ax[2].contourf(xi, yi, data_gridded['input'])
    for depth in test_definitions.test_definitions[test_name]['layers_pos']:
        ax[2].hlines(y=depth, xmin=xi.min(), xmax=xi.max(), linestyle='-', color='r')
    ax[2].set_title('Input model')
    ax[2] = slopestabilitytools.set_labels(ax[2])
    cb.append(plt.colorbar(im2, ax=ax[2], label='Resistivity log(ohm*m)'))  # , shrink=0.9)
    tick_locator = ticker.MaxNLocator(nbins=4)
    cb[2].locator = tick_locator
    cb[2].update_ticks()

    # Plot difference between correct and predicted classes
    im3 = ax[3].contourf(xi, yi, class_diff)
    for depth in test_definitions.test_definitions[test_name]['layers_pos']:
        ax[3].hlines(y=depth, xmin=xi.min(), xmax=xi.max(), linestyle='-', color='r')
    ax[3].set_title('Difference')
    ax[3] = slopestabilitytools.set_labels(ax[3])
    cb.append(plt.colorbar(im3, ax=ax[3], label='Is class correct?'))  # , shrink=0.9)
    tick_locator = ticker.MaxNLocator(nbins=2)
    cb[3].locator = tick_locator
    cb[3].update_ticks()

    # Plot sensitivity
    im4 = ax[4].contourf(xi, yi, data_gridded['sen'])
    for depth in test_definitions.test_definitions[test_name]['layers_pos']:
        ax[4].hlines(y=depth, xmin=xi.min(), xmax=xi.max(), linestyle='-', color='r')
    ax[4].set_title('Sensitivity')
    ax[4] = slopestabilitytools.set_labels(ax[4])
    cb.append(plt.colorbar(im4, ax=ax[4], label='Sensitivity'))  # , shrink=0.9)
    tick_locator = ticker.MaxNLocator(nbins=4)
    cb[4].locator = tick_locator
    cb[4].update_ticks()

    # Plot depth
    im5 = ax[5].contourf(xi, yi, data_gridded['depth'])
    ax[5].set_title('Depth')
    ax[5] = slopestabilitytools.set_labels(ax[5])
    cb.append(plt.colorbar(im5, ax=ax[5], label='Depth [m]'))  # , shrink=0.9)
    tick_locator = ticker.MaxNLocator(nbins=4)
    cb[5].locator = tick_locator
    cb[5].update_ticks()

    # Plot histogramm of input model
    ax[6].hist(data_gridded['input'])
    ax[6].set_title('Input histogramm')
    ax[6].set_xlabel('Value (Bin)')
    ax[6].set_ylabel('Count')

    # Plot histogramm of predicted classes
    ax[6].hist(data_gridded['input'])
    ax[6].set_title('Predicted classes histogramm')
    ax[6].set_xlabel('Value (Bin)')
    ax[6].set_ylabel('Count')

    fig.tight_layout()

    if training is True:
        slopestabilitytools.save_plot(fig, clf_name, '_ML_{}_class_overview_training'.format(test_name),
                                      subfolder='ML/training/overview')

    else:
        slopestabilitytools.save_plot(fig, clf_name, '_ML_{}_class_overview_prediction'.format(test_name),
                                      subfolder='ML/prediction/overview')

    return
