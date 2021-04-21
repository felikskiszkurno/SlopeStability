#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08.04.2021

@author: Feliks Kiszkurno
"""

import slostabcreatedata


def init(path):

    global test_parameters

    test_parameters = {}

    if path is not '':
        test_parameters = slostabcreatedata.read_test_parameters(path)
    else:
        print('temp')
    # test_definitions = {'hor1_01': {'layer_n': 1, 'rho_values': [[1, 5], [2, 15]], 'layers_pos': np.array([-5])},
    #                     'hor1_02': {'layer_n': 1, 'rho_values': [[1, 5], [2, 50]], 'layers_pos': np.array([-5])},
    #                     'hor1_03': {'layer_n': 1, 'rho_values': [[1, 15], [2, 20]], 'layers_pos': np.array([-8])},
    #                     'hor1_04': {'layer_n': 1, 'rho_values': [[1, 5], [2, 10]], 'layers_pos': np.array([-3])},
    #                     'hor1_05': {'layer_n': 1, 'rho_values': [[1, 5], [2, 25]], 'layers_pos': np.array([-3])},
    #                     'hor1_06': {'layer_n': 1, 'rho_values': [[1, 2], [2, 10]], 'layers_pos': np.array([-4])},
    #                     'hor1_07': {'layer_n': 1, 'rho_values': [[1, 10], [2, 20]], 'layers_pos': np.array([-6])},
    #                     'hor1_08': {'layer_n': 1, 'rho_values': [[1, 5], [2, 25]], 'layers_pos': np.array([-3])},
    #                     'hor1_09': {'layer_n': 1, 'rho_values': [[1, 3], [2, 25]], 'layers_pos': np.array([-3])},
    #                     'hor1_10': {'layer_n': 1, 'rho_values': [[1, 5], [2, 25]], 'layers_pos': np.array([-7])},
    #                     'hor1_11': {'layer_n': 1, 'rho_values': [[1, 10], [2, 12]], 'layers_pos': np.array([-4])},
    #                     'hor1_12': {'layer_n': 1, 'rho_values': [[1, 15], [2, 50]], 'layers_pos': np.array([-5])},
    #                     'hor1_14': {'layer_n': 1, 'rho_values': [[1, 5], [2, 75]], 'layers_pos': np.array([-5])},
    #                     'hor1_15': {'layer_n': 1, 'rho_values': [[1, 15], [2, 50]], 'layers_pos': np.array([-8])},
    #                     'hor1_16': {'layer_n': 1, 'rho_values': [[1, 25], [2, 50]], 'layers_pos': np.array([-5])},
    #                     'hor1_17': {'layer_n': 1, 'rho_values': [[1, 25], [2, 75]], 'layers_pos': np.array([-5])},
    #                     'hor1_18': {'layer_n': 1, 'rho_values': [[1, 5], [2, 75]], 'layers_pos': np.array([-8])},
    #                     'hor1_19': {'layer_n': 1, 'rho_values': [[1, 50], [2, 60]], 'layers_pos': np.array([-7])},
    #                     # 'hor1_20': {'layer_n': 1, 'rho_values': [[1, 2], [2, 10]], 'layers_pos': np.array([-15])},
    #
    #                     # Tests with two layers
    #                     # 'hor2_01': {'layer_n': 2, 'rho_values': [[1, 3], [2, 5], [3, 15]],
    #                     #             'layers_pos': np.array([-3, -6])},
    #                     # 'hor2_02': {'layer_n': 2, 'rho_values': [[1, 2], [2, 4], [3, 8]],
    #                     #             'layers_pos': np.array([-4, -8])},
    #                     # 'hor2_03': {'layer_n': 2, 'rho_values': [[1, 4], [2, 15], [3, 25]],
    #                     #             'layers_pos': np.array([-4, -8])},
    #                     # 'hor2_04': {'layer_n': 2, 'rho_values': [[1, 5], [2, 20], [3, 50]],
    #                     #             'layers_pos': np.array([-4, -8])},
    #                     # 'hor2_05': {'layer_n': 2, 'rho_values': [[1, 5], [2, 35], [3, 50]],
    #                     #             'layers_pos': np.array([-4, -8])},
    #                     # 'hor2_06': {'layer_n': 2, 'rho_values': [[1, 5], [2, 35], [3, 50]],
    #                     #             'layers_pos': np.array([-4, -8])},
    #                     # 'hor2_07': {'layer_n': 2, 'rho_values': [[1, 5], [2, 10], [3, 70]],
    #                     #             'layers_pos': np.array([-4, -8])},
    #                     # 'hor2_08': {'layer_n': 2, 'rho_values': [[1, 5], [2, 10], [3, 70]],
    #                     #             'layers_pos': np.array([-3, -9])},
    #                     # 'hor2_09': {'layer_n': 2, 'rho_values': [[1, 3], [2, 40], [3, 70]],
    #                     #             'layers_pos': np.array([-4, -8])},
    #                     # 'hor2_10': {'layer_n': 2, 'rho_values': [[1, 3], [2, 40], [3, 70]],
    #                     #             'layers_pos': np.array([-4, -8])},
    #                     # 'hor2_11': {'layer_n': 2, 'rho_values': [[1, 5], [2, 10], [3, 70]],
    #                     #             'layers_pos': np.array([-3, -15])},
    #                     # 'hor2_12': {'layer_n': 2, 'rho_values': [[1, 5], [2, 10], [3, 70]],
    #                     #             'layers_pos': np.array([-4, -6])},
    #                     # 'hor2_13': {'layer_n': 2, 'rho_values': [[1, 5], [2, 35], [3, 50]],
    #                     #             'layers_pos': np.array([-5, -20])},
    #                     # 'hor2_14': {'layer_n': 2, 'rho_values': [[1, 5], [2, 35], [3, 50]],
    #                     #             'layers_pos': np.array([-3, -5])},
    #                     # 'hor2_15': {'layer_n': 2, 'rho_values': [[1, 5], [2, 10], [3, 70]],
    #                     #             'layers_pos': np.array([-5, -15])},
    #                     # 'hor2_16': {'layer_n': 2, 'rho_values': [[1, 5], [2, 10], [3, 70]],
    #                     #             'layers_pos': np.array([-3, -15])},
    #                     # 'hor2_17': {'layer_n': 2, 'rho_values': [[1, 2], [2, 4], [3, 8]],
    #                     #             'layers_pos': np.array([-4, -15])},
    #                     # 'hor2_18': {'layer_n': 2, 'rho_values': [[1, 5], [2, 10], [3, 15]],
    #                     #             'layers_pos': np.array([-3, -15])},
    #                     # 'hor2_19': {'layer_n': 2, 'rho_values': [[1, 15], [2, 20], [3, 30]],
    #                     #             'layers_pos': np.array([-3, -15])},
    #                     # 'hor2_20': {'layer_n': 2, 'rho_values': [[1, 5], [2, 10], [3, 70]],
    #                     #             'layers_pos': np.array([-3, -10])}
    #                     }
