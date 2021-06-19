#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 18.06.2021

@author: Feliks Kiszkurno
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import slopestabilitytools


def plot_sen_corr(y_pred, y_answer, sen, clf_name, test_name, batch_name, *, training=False):

    class_diff = np.zeros_like(y_pred)
    class_diff[np.where(y_answer == y_pred)] = 1

    data = {'PRED': y_pred, 'COR': y_answer, 'SEN': sen, 'MATCH': class_diff}

    data_df = pd.DataFrame(data, columns=['PRED', 'COR', 'SEN', 'MATCH'])

    data_sort = data_df.sort_values(by=['MATCH', 'SEN'], ascending=False)

    sen_plot = data_sort['SEN'].to_numpy()
    match_plot = data_sort['MATCH'].to_numpy()

    fig, ax = plt.subplots(1, 1) 
    ax.plot(sen_plot, match_plot)
    fig.suptitle('Sensitivity vs correctness of classification')
    ax.set_xlabel('Sensitivity [normalized]')
    ax.set_ylabel('Correctness of classification [T/F]')

    if training is True:
        slopestabilitytools.save_plot(fig, clf_name, '_ML_{}_sen_cor'.format(test_name),
                                      subfolder='ML/training')

    else:
        slopestabilitytools.save_plot(fig, clf_name, '_ML_{}_sen_cor'.format(test_name),
                                      subfolder='ML/prediction', batch_name=batch_name)
