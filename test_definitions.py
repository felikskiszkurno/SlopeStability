#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08.04.2021

@author: Feliks Kiszkurno
"""

import numpy as np


def init():

    global test_definitions

    test_definitions = {'hor_01': {'layer_n': 1, 'rho_values': [[1, 5], [2, 15]], 'layers_pos': np.array([-5])},
                        'hor_02': {'layer_n': 1, 'rho_values': [[1, 5], [2, 50]], 'layers_pos': np.array([-5])},
                        'hor_03': {'layer_n': 1, 'rho_values': [[1, 15], [2, 20]], 'layers_pos': np.array([-8])},
                        'hor_04': {'layer_n': 1, 'rho_values': [[1, 5], [2, 10]], 'layers_pos': np.array([-3])},
                        'hor_05': {'layer_n': 1, 'rho_values': [[1, 5], [2, 25]], 'layers_pos': np.array([-3])},
                        'hor_06': {'layer_n': 1, 'rho_values': [[1, 2], [2, 10]], 'layers_pos': np.array([-4])},
                        'hor_07': {'layer_n': 1, 'rho_values': [[1, 10], [2, 20]], 'layers_pos': np.array([-6])},
                        'hor_08': {'layer_n': 1, 'rho_values': [[1, 5], [2, 25]], 'layers_pos': np.array([-3])},
                        'hor_09': {'layer_n': 1, 'rho_values': [[1, 3], [2, 25]], 'layers_pos': np.array([-3])},
                        'hor_10': {'layer_n': 1, 'rho_values': [[1, 5], [2, 25]], 'layers_pos': np.array([-7])},
                        'hor_11': {'layer_n': 1, 'rho_values': [[1, 10], [2, 12]], 'layers_pos': np.array([-4])},
                        'hor_12': {'layer_n': 1, 'rho_values': [[1, 15], [2, 50]], 'layers_pos': np.array([-5])},
                        'hor_13': {'layer_n': 2, 'rho_values': [[1, 3], [2, 5], [3, 15]],
                                   'layers_pos': np.array([-3, -6])},
                        'hor_14': {'layer_n': 2, 'rho_values': [[1, 2], [2, 4], [3, 8]],
                                   'layers_pos': np.array([-4, -8])},
                        'hor_15': {'layer_n': 2, 'rho_values': [[1, 4], [2, 15], [3, 25]],
                                   'layers_pos': np.array([-4, -8])},
                        'hor_16': {'layer_n': 2, 'rho_values': [[1, 5], [2, 20], [3, 50]],
                                   'layers_pos': np.array([-4, -8])}
                        }