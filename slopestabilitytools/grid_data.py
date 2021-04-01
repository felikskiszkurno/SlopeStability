#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 31.03.2021

@author: Feliks Kiszkurno
"""

import numpy as np
from scipy import interpolate
import math


def grid_data(x, y, data, *, regular_grid=False):  # data has to be a dictionary, output in dictionary in the same order as input

    x_min = math.ceil(np.min(x))
    x_max = math.floor(np.max(x))
    y_min = math.ceil(np.min(y))
    y_max = math.floor(np.max(y))
    if regular_grid is False:
        x_n = len(x)
        y_n = len(y)
    elif regular_grid is True:
        x_n = ((x_max - x_min) + 1) * 20
        y_n = ((y_max - y_min) + 1) * 20

    y_start = y_max
    y_end = y_min

    xi = np.linspace(x_min, x_max, x_n)
    yi = np.linspace(y_start, y_end, y_n)
    xx, yy = np.meshgrid(xi, yi)

    data_out = {}

    for key in data.keys():
        result_temp = interpolate.griddata((x, y), data[key], (xx, yy), method='nearest')
        data_out[key] = result_temp

    return xi, yi, data_out
