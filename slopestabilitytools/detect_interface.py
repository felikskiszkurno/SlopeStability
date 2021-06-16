#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 10.05.2021

@author: Feliks Kiszkurno
"""

import numpy as np
from scipy import interpolate


def detect_interface(x, y, y_pred_grid, *, continuity_threshold=30, separation_threshold=1.5):
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
        # print(str(value) + ': ' + str((np.count_nonzero(y_pred_grid_deri == value) / y_pred_col_num) * 100))
        if (np.count_nonzero(y_pred_grid_deri == value) / y_pred_col_num) * 100 > continuity_threshold:
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
            # print(ids_temp)
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
    '''
    # del inter_id
    #
    # # Combine the data in a dictonary
    # interface_potential_data = {}
    # for inter_id in range(len(potential_interfaces_values)):
    #     interface_potential_data[inter_id] = {
    #         'pot_value': potential_interfaces_values[inter_id],
    #         'depths': interface_depths[inter_id, :],
    #         'depth_mean': np.nanmean(interface_depths[inter_id, :]),
    #         'ids': interface_ids[inter_id, :],
    #         'param_a': interface_params_a[inter_id],
    #         'param_b': interface_params_b[inter_id]
    #     }
    #
    #
    # # Check if the layers should be combined
    # potential_interfaces_depths = interface_params_b
    # threshold = 1.5
    # interfaces_final = {}
    # interfaces_final_number = 0
    # if len(potential_interfaces_values) == 1:
    #     interfaces_final_number = 1
    #     inter_temp = interface_depths[0, :].copy()
    #     x_temp = x[~np.isnan(inter_temp)].reshape([x[~np.isnan(inter_temp)].size])
    #     y_temp = inter_temp[~np.isnan(inter_temp)].reshape([inter_temp[~np.isnan(inter_temp)].size])
    #     interfaces_final[interfaces_final_number] = {'x': x_temp, 'y': y_temp, 'depth': np.nanmean(y_temp)}
    # else:
    #     for inter_id in range(len(potential_interfaces_values) - 1):
    #         interfaces_final_number = interfaces_final_number + 1
    #         diff = abs(potential_interfaces_depths[inter_id] - potential_interfaces_depths[inter_id + 1])
    #         if diff < threshold:
    #             inter_temp = interface_depths[inter_id, :].copy()
    #             x1 = x[~np.isnan(inter_temp)].reshape([x[~np.isnan(inter_temp)].size])
    #             y1 = inter_temp[~np.isnan(inter_temp)].reshape([inter_temp[~np.isnan(inter_temp)].size])
    #
    #             inter_temp = interface_depths[inter_id + 1, :].copy()
    #             x2 = x[~np.isnan(inter_temp)].reshape([x[~np.isnan(inter_temp)].size])
    #             y2 = inter_temp[~np.isnan(inter_temp)].reshape([inter_temp[~np.isnan(inter_temp)].size])
    #
    #             interface_combined_depths = np.concatenate((y1, y2))
    #             interface_combined_x = np.concatenate((x1, x2))
    #
    #             #function = interpolate.interp1d(interface_combined_x, interface_combined_depths)
    #             #y_new = function(x)
    #
    #             interfaces_final[interfaces_final_number] = {'x': interface_combined_x,
    #                                                          'y': interface_combined_depths,
    #                                                          'depth': np.nanmean(interface_combined_depths)}
    #             inter_id = inter_id + 1
    #         else:
    #             inter_temp = interface_depths[inter_id, :].copy()
    #             x_temp = x[~np.isnan(inter_temp)].reshape([x[~np.isnan(inter_temp)].size])
    #             y_temp = inter_temp[~np.isnan(inter_temp)].reshape([inter_temp[~np.isnan(inter_temp)].size])
    #             interfaces_final[interfaces_final_number] = {'x': x_temp, 'y': y_temp, 'depth': np.nanmean(y_temp)}
    #
    #             if inter_id == len(potential_interfaces_values) - 1:
    #                 inter_id = inter_id + 1
    #                 interfaces_final_number = interfaces_final_number + 1
    #                 inter_temp = interface_depths[inter_id, :].copy()
    #                 x_temp = x[~np.isnan(inter_temp)].reshape([x[~np.isnan(inter_temp)].size])
    #                 y_temp = inter_temp[~np.isnan(inter_temp)].reshape([inter_temp[~np.isnan(inter_temp)].size])
    #                 interfaces_final[interfaces_final_number] = {'x': x_temp, 'y': y_temp, 'depth': np.nanmean(y_temp)}
    '''

    # Combine the data in a dictonary
    interface_potential_data = {}
    for inter_id in range(len(potential_interfaces_values)):
        interface_potential_data[inter_id] = {
            'pot_value': potential_interfaces_values[inter_id],
            'depths': interface_depths[inter_id, :],
            'depth_mean': np.nanmean(interface_depths[inter_id, :]),
            'ids': interface_ids[inter_id, :],
            'param_a': interface_params_a[inter_id],
            'param_b': interface_params_b[inter_id],
            'x': x
            # 'x': xi[~np.isnan(interface_depths[inter_id, 0:-1])].reshape([xi[~np.isnan(interface_depths[inter_id, 0:-1])].size])
        }

    # threshold = 1.5
    modified = True

    while modified is True:
        modified = False
        pairs_to_combine = []
        # print('a')
        # Find all pairs that are closer to each other than the threshold distance
        for interf_a_id in interface_potential_data.keys():
            # print('b')
            for interf_b_id in interface_potential_data.keys():
                # print('c')
                if interf_a_id != interf_b_id:
                    # print('d')
                    diff = abs(
                        interface_potential_data[interf_a_id]['depth_mean'] - interface_potential_data[interf_b_id][
                            'depth_mean'])
                    if diff <= separation_threshold:
                        # print('e')
                        modified = True
                        pairs_to_combine.append([interf_a_id, interf_b_id])

                else:
                    # print('f')
                    continue
        # Create new dictionary with interfaces
        paired_values = []
        for pair in pairs_to_combine:
            paired_values.extend(pair)

        interfaces_new = {}
        key_counter = 0
        for interf_id in interface_potential_data.keys():
            if inter_id not in paired_values:
                interfaces_new[key_counter] = interface_potential_data[interf_id]
                key_counter += 1

        for pair in pairs_to_combine:
            inter_a = pair[0]
            inter_b = pair[1]

            pot_value = [potential_interfaces_values[inter_a], potential_interfaces_values[inter_b]]
            depths_new = np.concatenate(
                (interface_potential_data[inter_a]['depths'], interface_potential_data[inter_b]['depths']))
            ids_new = np.concatenate(
                (interface_potential_data[inter_a]['ids'], interface_potential_data[inter_b]['ids']))
            param_a_new = (interface_potential_data[inter_a]['param_a'] + interface_potential_data[inter_b]['param_a'])/2
            param_b_new = (interface_potential_data[inter_a]['param_b'] + interface_potential_data[inter_b]['param_b'])/2
            x_new = np.concatenate((interface_potential_data[inter_a]['x'], interface_potential_data[inter_b]['x']))
            depths_new_orig = depths_new.copy()
            depths_new = depths_new[np.isfinite(depths_new)].reshape([depths_new[np.isfinite(depths_new)].size])
            x_new = x_new[np.isfinite(depths_new_orig)].reshape([x_new[np.isfinite(depths_new_orig)].size])
            # function_depth = interpolate.interp1d(x_new, depths_new, bounds_error=False, fill_value='extrapolate')
            # depths_new_interp = function_depth(x)
            function_depth = lambda a, b, x_d: a * x_d + b

            depths_new_interp = function_depth(param_a_new, param_b_new, x_new)

            interfaces_new[inter_id] = {
                'pot_value': pot_value,
                'depths': depths_new,
                'depth_mean': np.nanmean(depths_new_interp),
                'ids': ids_new,
                'param_a': param_a_new,
                'param_b': param_b_new,
                'x': x_new
            }

            del pot_value, depths_new, ids_new, param_b_new, param_a_new, x_new, function_depth, depths_new_interp

        interface_potential_data = interfaces_new

    return interface_potential_data
