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

is_success = slopestabilitytools.folder_structure.create_folder_structure()

number_of_tests = 10
rho_spread_factor = 1.5
rho_max = 150
layers_min = 1
layers_max = 3
min_depth = 1
max_depth = 10

tests_horizontal = slopestabilitytools.model_params(number_of_tests,
                                                    rho_spread_factor, rho_max,
                                                    layers_min, layers_max,
                                                    min_depth, max_depth)

for test_name in tests_horizontal.keys():
    test_result_curr = slostabcreatedata.create_data(test_name, tests_horizontal[test_name])

world_boundary_v = [-200, 0]  # [right, left border] relatively to the middle
world_boundary_h = [200, -100]  # [top, bottom border]


fig, ax = plt.subplots(3)
fig.suptitle(test_name)

pg.show(ert_manager.paraDomain, input_model2, ax=ax[0])
ax[0].set_title("Model on the output mesh")

pg.show(ert_manager.paraDomain, result, ax=ax[1])
ax[1].set_title("Inverted")

pg.show(ert_manager.paraDomain, result - input_model2, ax=ax[2])
ax[2].set_title("Diff")

fig.savefig('results/figs/hor_{}_results.eps'.format(test_name))
fig.savefig('results/figs/hor_{}_results.png'.format(test_name))

fig_input, ax_input = plt.subplots(1)
pg.show(mesh, input_model, ax=ax_input)
ax_input.set_title('1 Geometry of the model')

fig_input.savefig('results/figs/hor_{}_input.eps'.format(test_name))
fig_input.savefig('results/figs/hor_{}_input.png'.format(test_name))
