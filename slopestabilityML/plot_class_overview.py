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
import test_definitions
import settings


def plot_class_overview(test_results, test_name, class_in, y_pred, clf_name, *, training=False, depth_estimate='x', interface_y='x', interface_x='x', depth_accuracy='x'):

    x = test_results['X'].to_numpy()
    y = test_results['Y'].to_numpy()

    class_in = class_in.to_numpy()
    class_in = class_in.reshape(class_in.size)

    # Create plot
    fig, _ax = plt.subplots(nrows=4, ncols=2, figsize=(1.35*10, 10))
    ax = _ax.flatten()
    cb = []

    #fig.suptitle('Classification overview: ' + test_name + ', ' + clf_name + ', depth estimate accuracy: ' + str(depth_accuracy) + '%, depth (est/true): ' + str(depth_estimate) + '/' + str(test_definitions.test_parameters[test_name]['layers_pos'][0]))
    depth_true = test_definitions.test_parameters[test_name]['layers_pos'][0]
    fig.suptitle('Classification overview: {}, {}, depth estimate RMSE: {:.2f}%, depth (est/true): {:.2f}/{:.2f}'.format(test_name, clf_name, depth_accuracy, depth_estimate, depth_true))
    fig.subplots_adjust(hspace=0.8)

    # Convert labels to numerical for plotting
    if settings.settings['use_labels'] is True:
        class_in = slopestabilitytools.label2numeric(class_in)

    # Plot input classes
    im0 = ax[0].scatter(x, y, c=class_in)
    for depth in test_definitions.test_parameters[test_name]['layers_pos']:
        ax[0].hlines(y=depth, xmin=x.min(), xmax=x.max(), linestyle='-', color='r')
    ax[0].set_title('Input classes')
    ax[0] = slopestabilitytools.set_labels(ax[0])
    cb.append(plt.colorbar(im0, ax=ax[0], label='Class'))  # , shrink=0.9)
    tick_locator = ticker.MaxNLocator(nbins=4)
    cb[0].locator = tick_locator
    cb[0].update_ticks()

    # Convert labels to numerical for plotting
    if settings.settings['use_labels'] is True:
        y_pred = slopestabilitytools.label2numeric(y_pred)

    # Plot prediction
    im1 = ax[1].scatter(x, y, c=y_pred)
    for depth in test_definitions.test_parameters[test_name]['layers_pos']:
        ax[1].hlines(y=depth, xmin=x.min(), xmax=x.max(), linestyle='-', color='r')
    ax[1].hlines(y=depth_estimate, xmin=x.min(), xmax=x.max(), linestyle='-', color='g')
    ax[1].plot(interface_x, interface_y)
    ax[1].set_title('Predicted classes')
    ax[1] = slopestabilitytools.set_labels(ax[1])
    cb.append(plt.colorbar(im1, ax=ax[1], label='Class'))  # , shrink=0.9)
    tick_locator = ticker.MaxNLocator(nbins=4)
    cb[1].locator = tick_locator
    cb[1].update_ticks()

    # Plot input model
    im2 = ax[2].scatter(x, y, c=test_results['INMN'].to_numpy())
    for depth in test_definitions.test_parameters[test_name]['layers_pos']:
        ax[2].hlines(y=depth, xmin=x.min(), xmax=x.max(), linestyle='-', color='r')
    ax[2].hlines(y=depth_estimate, xmin=x.min(), xmax=x.max(), linestyle='-', color='g')
    ax[2].set_title('Input model')
    ax[2] = slopestabilitytools.set_labels(ax[2])
    cb.append(plt.colorbar(im2, ax=ax[2], label='Resistivity log(ohm*m)'))  # , shrink=0.9)
    tick_locator = ticker.MaxNLocator(nbins=4)
    cb[2].locator = tick_locator
    cb[2].update_ticks()

    class_diff = np.zeros_like(y_pred)
    class_diff[np.where(class_in == y_pred)] = 1

    # Plot difference between correct and predicted classes
    im3 = ax[3].scatter(x, y, c=class_diff)
    for depth in test_definitions.test_parameters[test_name]['layers_pos']:
        ax[3].hlines(y=depth, xmin=x.min(), xmax=x.max(), linestyle='-', color='r')
    ax[3].set_title('Difference')
    ax[3] = slopestabilitytools.set_labels(ax[3])
    cb.append(plt.colorbar(im3, ax=ax[3], label='Is class correct?'))  # , shrink=0.9)
    tick_locator = ticker.MaxNLocator(nbins=2)
    cb[3].locator = tick_locator
    cb[3].update_ticks()

    # Plot sensitivity
    im4 = ax[4].scatter(x, y, c=test_results['SEN'].to_numpy())
    for depth in test_definitions.test_parameters[test_name]['layers_pos']:
        ax[4].hlines(y=depth, xmin=x.min(), xmax=x.max(), linestyle='-', color='r')
    ax[4].set_title('Sensitivity')
    ax[4] = slopestabilitytools.set_labels(ax[4])
    cb.append(plt.colorbar(im4, ax=ax[4], label='Sensitivity'))  # , shrink=0.9)
    tick_locator = ticker.MaxNLocator(nbins=4)
    cb[4].locator = tick_locator
    cb[4].update_ticks()

    # Plot depth
    im5 = ax[5].scatter(x, y, c=y)
    ax[5].set_title('Depth')
    ax[5] = slopestabilitytools.set_labels(ax[5])
    cb.append(plt.colorbar(im5, ax=ax[5], label='Depth [m]'))  # , shrink=0.9)
    tick_locator = ticker.MaxNLocator(nbins=4)
    cb[5].locator = tick_locator
    cb[5].update_ticks()

    # Plot histogramm of input model
    ax[6].hist(test_results['INMN'].to_numpy())
    ax[6].set_title('Input histogramm')
    ax[6].set_xlabel('Value (Bin)')
    ax[6].set_ylabel('Count')

    # Plot histogramm of predicted classes
    ax[7].hist(y_pred)
    ax[7].set_title('Predicted classes histogramm')
    ax[7].set_xlabel('Value (Bin)')
    ax[7].set_ylabel('Count')

    fig.tight_layout()

    if training is True:
        slopestabilitytools.save_plot(fig, clf_name, '_ML_{}_class_overview_training'.format(test_name),
                                      subfolder='ML/training')

    else:
        slopestabilitytools.save_plot(fig, clf_name, '_ML_{}_class_overview_prediction'.format(test_name),
                                      subfolder='ML/prediction')

    return
