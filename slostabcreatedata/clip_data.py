#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30.03.2021

@author: Feliks Kiszkurno
"""

import numpy as np


def clip_data(array, array_max, array_min):

    ids_max = np.argwhere(array > array_max)
    ids_min = np.argwhere(array < array_min)

    array[ids_max] = array_max
    array[ids_min] = array_min

    return array
