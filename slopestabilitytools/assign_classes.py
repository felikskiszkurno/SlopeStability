#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 21.03.2021

@author: Feliks Kiszkurno
"""

import numpy as np
import settings
import slopestabilitytools.def_classes


# It assumes, that input array is normalized

def assign_classes(data_array):

    classes_def = slopestabilitytools.def_classes(settings.settings['norm_class_num'])
    # classes_def = np.array([[0, 0, 0.1],
    #                         [1, 0.1, 0.2],
    #                         [2, 0.2, 0.3],
    #                         [3, 0.3, 0.4],
    #                         [4, 0.4, 0.5],
    #                         [5, 0.5, 0.6],
    #                         [6, 0.6, 0.7],
    #                         [7, 0.7, 0.8],
    #                         [8, 0.8, 0.9],
    #                         [9, 0.9, 1.0]])
    #
    # classes_def = np.array([[0, 0, 0.2],
    #                         [1, 0.2, 0.4],
    #                         [2, 0.4, 0.6],
    #                         [3, 0.6, 0.8],
    #                         [4, 0.8, 1.0]])

    # TODO: maybe reshaping is not necessary, as long as I supply the input as a vector. Think about and change it.
    classes = np.zeros_like(data_array)
    classes_shape = classes.shape
    classes = classes.reshape([classes.size])

    for class_pair in classes_def:
        # print(class_pair)
        ind = np.argwhere(np.logical_and(data_array > class_pair[1], data_array <= class_pair[2]))
        # print(ind)
        classes[ind] = class_pair[0]

    classes_array = classes.reshape(classes_shape)

    return classes_array
