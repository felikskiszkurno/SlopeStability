#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19.01.2021

@author: Feliks Kiszkurno
"""

import slopestabilitytools
import random
import math
import settings


def split_dataset(test_names, random_seed, *, proportion=False):

    if proportion is False:
        proportion = settings.settings['split_proportion']

    random.seed(random_seed)

    test_number = len(test_names)
    test_prediction = random.sample(list(test_names),
                                     k=math.ceil(test_number * proportion))

    test_training = slopestabilitytools.set_diff(list(test_names), set(test_prediction))

    return sorted(test_training), sorted(test_prediction)
