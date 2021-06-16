#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 03.06.2021

@author: Feliks Kiszkurno
"""

import matplotlib.pyplot as plt
import pandas as pd
import os
import settings


def plot_feature_importance(clf, importance, x_train, test_name, *, batch_name=''):
    # importance = permutation_importance(clf, x_train, y_question)

    importance_mean_norm = importance.importances_mean / sum(importance.importances_mean)

    importance_df = pd.DataFrame(importance_mean_norm,
                                 columns=['Coefficients'],
                                 index=x_train.columns
                                 )

    fig = importance_df.plot(kind='barh')
    plt.title(test_name + ' feature importance')
    plt.axvline(x=0, color='.5')
    plt.subplots_adjust(left=.3)

    for file_format in settings.settings['plot_formats']:
        figure_name = clf.steps[1][0] + '_' + test_name + '.' + file_format


        fig.figure.savefig(os.path.join(settings.settings['figures_folder'], batch_name, 'ML', 'feature_importance',
                                    file_format, figure_name))

