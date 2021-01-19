#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19.01.2021

@author: Feliks Kiszkurno
"""

import slopestabilitytools
import random
import math


def split_dataset(test_names, random_seed):

    random.seed(random_seed)

    test_number = len(test_names)
    test_prediction = random.choices(list(test_names),
                                     k=math.ceil(test_number * 0.1))

    test_training = slopestabilitytools.set_diff(list(test_names), set(test_prediction))

    return test_training, test_prediction
