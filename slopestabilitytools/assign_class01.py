#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 21.03.2021

@author: Feliks Kiszkurno
"""

import numpy as np


def assign_class01(input_model, resistivity_map):

    classes = []
    resistivity_values = []

    for pair in resistivity_map:
        resistivity_values.append(pair[1])

    for value in input_model:
        res_diff = np.abs(value * np.ones_like(resistivity_values) - resistivity_values)
        res_index = np.argmin(res_diff)
        classes.append(res_index)

    return classes
