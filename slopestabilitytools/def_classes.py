#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 26.03.2021

@author: Feliks Kiszkurno
"""

import numpy as np


def def_classes(class_num):

    class_spacing = 1 / class_num
    ids = np.linspace(0, class_num - 1, class_num)
    class_start = np.linspace(0, 1 - class_spacing, class_num)
    class_end = np.linspace(class_spacing, 1, class_num)
    classes_def = np.vstack((ids, class_start, class_end))

    return classes_def.T