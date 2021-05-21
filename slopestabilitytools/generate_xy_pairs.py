#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 21.05.2021

@author: Feliks Kiszkurno
"""

import numpy as np


def generate_xy_pairs(x_values, y_values):

    x_new = []
    y_new = []

    for x in x_values:
        x_new.extend(x*np.ones([y_values.size]))
        y_new.extend(y_values)

    return x_new, y_new
