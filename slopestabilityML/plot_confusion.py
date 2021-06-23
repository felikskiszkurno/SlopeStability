#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22.06.2021

@author: Feliks Kiszkurno
"""

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import slopestabilitytools
import settings


def plot_confusion(clf_name, clf, *, summary=False, y_pred=0, y_true=0, conf_mat=0,
                   test_name='', batch_name='', training=False):

    classes = np.arange(settings.settings['norm_class_num'])

    conf_matrix = np.zeros([len(classes), len(classes)])
    # if summary is False:
    #     conf_matrix = confusion_matrix(y_true, y_pred, labels=classes)
    # else:
    #     conf_matrix = conf_mat

    disp = plot_confusion_matrix(clf, y_pred, y_true,
                                 labels=classes,
                                 display_labels=classes, normalize='all')

    disp.figure_.suptitle(test_name+' confusion matrix')

    if training is True:
        slopestabilitytools.save_plot(disp.figure_, clf_name, '_ML_{}_confusion_matrix_training'.format(test_name),
                                      subfolder='ML/training', batch_name=batch_name)

    else:
        slopestabilitytools.save_plot(disp.figure_, clf_name, '_ML_{}_confusion_matrix_prediction'.format(test_name),
                                      subfolder='ML/prediction', batch_name=batch_name)

    return conf_matrix
