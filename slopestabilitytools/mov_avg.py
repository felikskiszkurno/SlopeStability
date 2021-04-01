#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 31.03.2021

@author: Feliks Kiszkurno
"""

import numpy as np


def mov_avg(array, window_length, *, method='same'):
    array_avg = np.convolve(array, np.ones(window_length), method) / window_length
    return array_avg
