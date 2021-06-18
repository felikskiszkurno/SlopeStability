#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 18.06.2021

@author: Feliks Kiszkurno
"""

import numpy as np
import slopestabilitytools
import matplotlib.pyplot as plt


def plot_class_inv(class_result, ert_manager, test_name, plot_title):

    fig, _ax = plt.subplots(nrows=2, ncols=1)
    ax = _ax.flatten()

    fig.suptitle(test_name + plot_title + '_classes')
    fig.subplots_adjust(hspace=0.8)

    ax[0].hist(class_result)
    ax[0].set_title('Histogramm of classes assigned to inverted profile')
    ax[0].set_xlabel('Value (Bin)')
    ax[0].set_ylabel('Count')

    im1 = ax[1].scatter(ert_manager.paraDomain.cellCenters().array()[:, 0],
                        ert_manager.paraDomain.cellCenters().array()[:, 1], c=class_result)
    ax[1].set_title('Classes assigned to inverted profile')
    ax[1] = slopestabilitytools.set_labels(ax[1])

    fig.tight_layout()
    slopestabilitytools.save_plot(fig, test_name, plot_title + '_classes')
