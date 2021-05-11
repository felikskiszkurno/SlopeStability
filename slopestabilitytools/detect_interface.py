#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 10.05.2021

@author: Feliks Kiszkurno
"""

import numpy as np
from scipy import interpolate


def detect_interface(x, y, y_pred_grid):

    depths = y

    y_pred_col_num = y_pred_grid.shape[1]
    y_pred_grid_copy = y_pred_grid.copy()
    y_pred_uniq = np.unique(y_pred_grid)

    # Assign new values for each class
    y_pred_new_values = np.zeros(len(y_pred_uniq))

    for number in np.arange(1, len(y_pred_uniq), 1):
        y_pred_new_values[number] = y_pred_new_values[number - 1] + number

    for id_num in range(len(y_pred_uniq)):
        y_pred_grid_copy[y_pred_grid_copy == y_pred_uniq[id_num]] = y_pred_new_values[id_num]

    # Get "derivative" of the matrix
    result_grid_rolled = np.roll(y_pred_grid_copy, -1, axis=0)
    y_pred_grid_deri = y_pred_grid_copy - result_grid_rolled
    y_pred_grid_deri[-1, :] = 0
    potential_interfaces_values = []
    for value in np.unique(y_pred_grid_deri):
        #print(str(value) + ': ' + str((np.count_nonzero(y_pred_grid_deri == value) / y_pred_col_num) * 100))
        if (np.count_nonzero(y_pred_grid_deri == value) / y_pred_col_num) * 100 > 50:
            if value != 0:
                potential_interfaces_values.append(value)

    # Obtain depths and ids for potential interfaces
    interface_ids = np.zeros([len(potential_interfaces_values), y_pred_col_num]) * float("NaN")
    interface_depths = np.zeros([len(potential_interfaces_values), y_pred_col_num]) * float("NaN")
    for inter_id in range(len(potential_interfaces_values)):
        for col_id in range(y_pred_col_num):
            ids_temp = np.where(y_pred_grid_deri[:, col_id] == potential_interfaces_values[inter_id])
            if ids_temp[0].size > 1:
                ids_temp = int(np.mean(ids_temp))
            elif ids_temp[0].size == 0:
                continue
            #print(ids_temp)
            # No idea why np.where returns array and sometimes integer
            if isinstance(ids_temp, int) is True:
                interface_ids[inter_id, col_id] = ids_temp
                interface_depths[inter_id, col_id] = depths[ids_temp]
            else:
                interface_ids[inter_id, col_id] = ids_temp[0][0]
                interface_depths[inter_id, col_id] = depths[ids_temp[0][0]]

    # Obtain line parameterts
    interface_params_a = np.zeros([len(potential_interfaces_values)]) * float('NaN')
    interface_params_b = np.zeros([len(potential_interfaces_values)]) * float('NaN')
    for inter_id in range(len(potential_interfaces_values)):
        inter_temp = interface_depths[inter_id, :].copy()
        x_temp = x[~np.isnan(inter_temp)].reshape([x[~np.isnan(inter_temp)].size])
        y_temp = inter_temp[~np.isnan(inter_temp)].reshape([inter_temp[~np.isnan(inter_temp)].size])
        par = np.polyfit(x_temp, y_temp, 1, full=True)
        interface_params_a[inter_id] = par[0][0]
        interface_params_b[inter_id] = par[0][1]

    # Check if the layers should be combined
    threshold = 1.5
    interfaces_final = {}
    interfaces_final_number = 0
    for inter_id in range(len(potential_interfaces_values) - 1):
        interfaces_final_number = interfaces_final_number + 1
        diff = abs(interface_params_b[inter_id] - interface_params_b[inter_id + 1])
        if diff < threshold:
            inter_temp = interface_depths[inter_id, :].copy()
            x1 = x[~np.isnan(inter_temp)].reshape([x[~np.isnan(inter_temp)].size])
            y1 = inter_temp[~np.isnan(inter_temp)].reshape([inter_temp[~np.isnan(inter_temp)].size])

            inter_temp = interface_depths[inter_id + 1, :].copy()
            x2 = x[~np.isnan(inter_temp)].reshape([x[~np.isnan(inter_temp)].size])
            y2 = inter_temp[~np.isnan(inter_temp)].reshape([inter_temp[~np.isnan(inter_temp)].size])

            interface_combined_depths = np.concatenate((y1, y2))
            interface_combined_x = np.concatenate((x1, x2))

            #function = interpolate.interp1d(interface_combined_x, interface_combined_depths)
            #y_new = function(x)

            interfaces_final[interfaces_final_number] = {'x': interface_combined_x,
                                                         'y': interface_combined_depths,
                                                         'depth': np.nanmean(interface_combined_depths)}
            inter_id = inter_id + 1
        else:
            inter_temp = interface_depths[inter_id, :].copy()
            x_temp = x[~np.isnan(inter_temp)].reshape([x[~np.isnan(inter_temp)].size])
            y_temp = inter_temp[~np.isnan(inter_temp)].reshape([inter_temp[~np.isnan(inter_temp)].size])



            interfaces_final[interfaces_final_number] = {'x': x_temp, 'y': y_temp, 'depth': np.nanmean(y_temp)}

    return interfaces_final
