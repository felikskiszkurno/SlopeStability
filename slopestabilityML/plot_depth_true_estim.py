#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 25.05.2021

@author: Feliks Kiszkurno
"""

import matplotlib.pyplot as plt
import numpy as np
import slopestabilitytools


def plot_depth_true_estim(ml_results, *, batch_name=''):

    fig, ax = plt.subplots()

    colors = ['red', 'green', 'orange', 'blue', 'yellow', 'cyan', 'black', 'purple', 'khaki', 'orange', 'gold',
              'turquoise', 'orangered']
    colors_count = 0

    for classifier in ml_results.keys():
        if classifier is not 'com':

            depth_estim = []
            depth_true = []

            for depth_estim_value in ml_results[classifier]['prediction'][batch_name]['depth_estim']:

                if isinstance(depth_estim_value, list):

                    for value in depth_estim_value:

                        depth_estim.append(value)

                else:

                    depth_estim.append(value[0])

            for depth_true_value in ml_results[classifier]['prediction'][batch_name]['depth_true']:

                if isinstance(depth_true_value, list):

                    for value in depth_true_value:

                        depth_true.append(value)

                else:

                    depth_true.append(value[0])

            ax.plot(depth_estim, depth_true, marker='o', color=colors[colors_count], label=classifier, linestyle='None')
            colors_count += 1

    y_lim = ax.get_ylim()
    x_lim = ax.get_xlim()

    if y_lim[1] > x_lim[1]:
        ax_max = y_lim[1]
    else:
        ax_max = x_lim[1]

    if y_lim[0] > x_lim[0]:
        ax_min = y_lim[0]
    else:
        ax_min = x_lim[0]

    ax.set_xlim([ax_min, ax_max])
    ax.set_ylim([ax_min, ax_max])

    ref_x =np.arange(ax_min, ax_max, 1)
    ref_y = ref_x

    ax.plot(ref_x, ref_y, color='black', label='reference')

    plt.ylabel('True depth [m]')
    plt.xlabel('Predicted depth [m]')

    ax.legend()
    plt.title('Predicted vs True interface depth')

    slopestabilitytools.save_plot(fig, 'All', '_true_vs_pred', subfolder='ML/', batch_name=batch_name)
