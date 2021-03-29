#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15.03.2021

@author: Feliks Kiszkurno
"""

import numpy as np


def normalize(array):
    array_max = 0
    array = np.array(array)
    if array.ndim == 1:
        array_max = np.max(array)

    elif array.ndim == 2:
        array_max = np.amax(array)

    else:
        print('ERROR: incorrect dimensionality of the input array')
        exit()

    array_norm = array / array_max

    return array_norm

