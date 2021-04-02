#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 31.03.2021

@author: Feliks Kiszkurno
"""
import numpy as np
from scipy import interpolate
import math

import slopestabilitytools
import settings


def max_grad_classi(test_result):
    x = test_result['X']
    y = test_result['Y']

    if settings.settings['norm'] is True:
        result = test_result['RESN']
    elif settings.settings['norm'] is False:
        result = test_result['RES']

    data = {'result': result}

    xi, yi, data_gridded = slopestabilitytools.grid_data(x, y, data, regular_grid=True)

    result_grid = data_gridded['result']

    gradient = np.gradient(result_grid, axis=0)
    gradient2 = np.gradient(gradient, axis=0)
    gradient2 = gradient2 / np.amax(gradient2)
    ind = gradient2 < 0.15
    gradient2[ind] = 0

    inds = np.zeros(gradient2.T.shape[0])
    classes = np.ones_like(gradient2.T)

    for num, column in enumerate(gradient2.T):
        # print(num)
        inds[num] = np.argmax(column)
    inds = slopestabilitytools.mov_avg(inds, math.ceil(len(inds) / 10))
    for num, column in enumerate(classes):
        column[:int(inds[num])] = 0
    classes = classes.T

    classes_interp_f = interpolate.interp2d(xi, yi, classes)
    classes_interp = np.zeros_like(x)
    for cell_id in range(len(x)):
        classes_interp[cell_id] = classes_interp_f(x[cell_id], y[cell_id])

    return classes_interp
