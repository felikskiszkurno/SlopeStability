#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 10:29:00 2021

@author: felikskrno
"""

import numpy as np
from numpy import random

random.seed(999)


# number_of_tests = 1
# rho_spread_factor = 1.5
# max_rho = 150
# layers_min = 1
# layers_max = 3
# world_boundary_v = [-100, 0]  # [right, left border] relatively to the middle
# world_boundary_h = [200, -200]  # [top, bottom border]


def model_params(n_of_tests, rho_spread, rho_max, layers_n_min, layers_n_max, depth_min, depth_max):
    test_names = {}
    test_n_layers = {}
    test_rho = {}
    test_layers_pos = {}
    tests_horizontal = {}

    for test_id in range(n_of_tests):

        test_names[test_id] = 'hor_{}'.format(str(test_id + 1))
        test_n_layers[test_names[test_id]] = np.random.randint(layers_n_min, layers_n_max)
        rho_temp = []
        rho_used = []
        layer_pos_temp = []
        layer_pos_used = []
        layer = 0

        while layer < test_n_layers[test_names[test_id]]:

            new_layer_pos = random.randint(depth_min, depth_max)

            if len(layer_pos_temp) == 0:

                layer_pos_temp.append(new_layer_pos)
                layer_pos_used.append(new_layer_pos)

            else:

                if new_layer_pos not in layer_pos_used:

                    new_layer_pos = random.randint(depth_min, depth_max)
                    layer_pos_temp.append(new_layer_pos)
                    layer_pos_used.append(new_layer_pos)

                else:

                    while new_layer_pos in layer_pos_used:
                        new_layer_pos = random.randint(depth_min, depth_max)

                    layer_pos_temp.append(new_layer_pos)
                    layer_pos_used.append(new_layer_pos)

            layer += 1

        layer = 0

        while layer < test_n_layers[test_names[test_id]] + 1:

            new_rho = 0

            while new_rho == 0:

                new_rho = int(random.rand(1)[0] * rho_max)

            if len(rho_temp) == 0:

                # rho_temp.append([layer + 1, new_rho])
                rho_used.append(new_rho)

            else:

                if new_rho not in rho_used:

                    new_rho = int(random.rand(1)[0] * rho_max)
                    # rho_temp.append([layer + 1, new_rho])
                    rho_used.append(new_rho)

                else:

                    while new_rho in rho_used:
                        new_rho = int(random.rand(1)[0] * rho_max)

                    # rho_temp.append([layer + 1, new_rho])
                    rho_used.append(new_rho)

            layer += 1

        rho_temp = np.sort(rho_used)
        rho_final = []
        layer_id = 1

        for rho in rho_temp:
            rho_final.append([layer_id, rho])
            layer_id += 1

        test_layers_pos[test_names[test_id]] = -1 * (np.sort(np.array(layer_pos_temp)))  # np.flip(np.sort(np.array(layer_pos_temp)))
        test_rho[test_names[test_id]] = rho_final  # np.sort(rho_temp)
        rho_temp = []
        rho_used = []
        layer_pos = []
        layer_pos_temp = []
        layer_pos_used = []

        tests_horizontal.update({test_names[test_id]: {'layer_n': test_n_layers[test_names[test_id]],
                                                        'rho_values': rho_final,
                                                        'layers_pos': test_layers_pos[test_names[test_id]]}})

    #tests_horizontal = {'names': test_names,
         #               'layer_n': test_n_layers,
          #              'rho_values': test_rho,
            #            'layers_pos': test_layers_pos}

    return tests_horizontal
