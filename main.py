#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 10:29:00 2021

@author: felikskrno
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import slopestabilitytools
import slopestabilityML
import slostabcreatedata


import pygimli as pg
import pygimli.meshtools as mt
import pygimli.physics.ert as ert

# Prepare folder structure for output
is_success = slopestabilitytools.folder_structure.create_folder_structure()

# Settings
number_of_tests = 2
rho_spread_factor = 1.5
rho_max = 150
layers_min = 1
layers_max = 5
min_depth = 1
max_depth = 25

# Generate parameters for tests
tests_horizontal = slopestabilitytools.model_params(number_of_tests,
                                                    rho_spread_factor, rho_max,
                                                    layers_min, layers_max,
                                                    min_depth, max_depth)

#  Create models and invert them
test_results = {}

for test_name in tests_horizontal.keys():

    test_result_curr = slostabcreatedata.create_data(test_name, tests_horizontal[test_name], max_depth)
    test_results.update({test_name: test_result_curr})
    del test_result_curr

    # Plot and save figures
    slopestabilitytools.plot_and_save(test_name, test_results[test_name], 'Test: '+test_name)


slopestabilityML.svm_run(test_results)
