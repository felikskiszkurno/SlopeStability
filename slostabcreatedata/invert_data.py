#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14.06.2021

@author: Feliks Kiszkurno
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pygimli as pg
import pygimli.meshtools as mt
import pygimli.physics.ert as ert

import slopestabilitytools
import slostabcreatedata
import settings
import os


def invert_data(profile_name, *, lambda_param=20, z_weight_param=0.2):

    # Load data
    ert_manager = ert.ERTManager(os.path.join(settings.settings['data_measurement'], profile_name+'.ohm'),
                                 useBert=True, verbose=True, debug=False)

    # RUN INVERSION
    #k0 = pg.physics.ert.createGeometricFactors(data)

    model_inverted = ert_manager.invert(lam=lambda_param, paraDX=0.25, paraMaxCellSize=2, zWeight=z_weight_param,  # paraDepth=2 * max_depth,
                                        quality=34, zPower=0.4)

    result_full = ert_manager.inv.model

    result_array = result_full.array()

    limits = pg.utils.interperc(ert_manager.inv.model, trimval=25.0)
    limits = [50, 2000]
    result_array[result_array <= limits[0]] = limits[0]
    result_array[result_array >= limits[1]] = limits[1]

    resistivity_map = []
    resistivity_map.append([1, min(result_array)])
    resistivity_map.append([2, max(result_array)])

    fig_result, ax_result = plt.subplots(1)
    pg.show(ert_manager.paraDomain, result_array, label=pg.unit('res'), showMesh=True, ax=ax_result)
    ax_result = slopestabilitytools.set_labels(ax_result)
    ax_result.set_title('3 Result')
    fig_result.tight_layout()
    slopestabilitytools.save_plot(fig_result, profile_name, '_3_result')

    classes = slopestabilitytools.assign_class01(result_array, resistivity_map)

    # classesn = slopestabilitytools.assign_classes(slopestabilitytools.normalize(input_model2_array))
    classesn = slopestabilitytools.assign_classes(slopestabilitytools.normalize(np.log10(result_array)))

    # Create sensitivity values
    jac = ert_manager.fop.jacobian()  #

    # Coverage = cumulative sensitivity = all measurements
    cov = ert_manager.coverage()
    # pg.show(ert_manager.paraDomain, cov, label="Logarithm of cumulative sensitivity")

    rho_arr = []
    for entry in resistivity_map:
        rho_arr.append(entry[1])
    rho_arr = np.array(rho_arr)
    rho_max = np.max(rho_arr)
    rho_min = np.min(rho_arr)

    # result_array_norm = np.log10(result_array)
    result_array_norm = slopestabilitytools.normalize(np.log10(result_array))

    labels = slopestabilitytools.classes_labels.numeric2label(classesn)

    input_model = np.array([1]*len(result_array))

    profil_name_column = [profile_name] * len(input_model)

    # Results on unstructured grid
    experiment_results = pd.DataFrame(data={'NAME': profil_name_column,
                                            'X': ert_manager.paraDomain.cellCenters().array()[:, 0],
                                            'Y': ert_manager.paraDomain.cellCenters().array()[:, 1],
                                            'Z': ert_manager.paraDomain.cellCenters().array()[:, 2],
                                            'INM': input_model,
                                            'INMN': input_model,
                                            'RES': result_array,
                                            'RESN': result_array_norm,
                                            'SEN': cov,
                                            'CLASS': classes,
                                            'CLASSN': classesn,
                                            'LABELS': labels})
    experiment_results = experiment_results[(experiment_results.INMN > 0) & (experiment_results.RESN > 0)]

    experiment_results.to_csv(settings.settings['data_folder'] + '/' + profile_name + '.csv')

    plot_title = '_in_inv_diff'
    if settings.settings['norm_class'] is True:
        slopestabilitytools.plot_and_save_pg(profile_name, plot_title, ert_manager,
                                             input_model, result_full, classesn)
        slopestabilitytools.plot_class_inv(classesn, ert_manager, profile_name, plot_title)
    else:
        slopestabilitytools.plot_and_save_pg(profile_name, plot_title, ert_manager,
                                             input_model, result_full, classes)
        slopestabilitytools.plot_class_inv(classes, ert_manager, profile_name, plot_title)

    if settings.settings['grd'] is True:
        # Results on structured grid
        x_min = np.ceil(np.min(ert_manager.paraDomain.cellCenters().array()[:, 0]))
        x_max = np.floor(np.max(ert_manager.paraDomain.cellCenters().array()[:, 0]))
        x_values = np.arange(x_min, x_max, settings.settings['resample_x_spacing'])

        y_min = np.ceil(np.min(ert_manager.paraDomain.cellCenters().array()[:, 1]))
        y_max = np.ceil(np.max(ert_manager.paraDomain.cellCenters().array()[:, 1]))
        y_values = np.arange(y_min, y_max, settings.settings['resample_y_spacing'])

        z_values = np.zeros([len(y_values)])
        # x_grid, y_grid = slopestabilitytools.generate_xy_pairs(x_values, y_values)

        grid = pg.createGrid(x=x_values, y=y_values)

        input_model2_grd = pg.interpolate(srcMesh=ert_manager.paraDomain, inVec=input_model,
                                          destPos=grid.cellCenters())

        input_model2_grd_array = input_model2_grd.array()

        profil_name_column_grd = [profile_name] * len(input_model2_grd_array)

        input_model2_grd_array_norm = np.log10(input_model2_grd)

        result_grd = pg.interpolate(srcMesh=ert_manager.paraDomain, inVec=result_full, destPos=grid.cellCenters())

        result_grd_array = result_grd.array()

        result_grd_array_norm = np.log10(result_grd_array)

        cov_grd = pg.interpolate(srcMesh=ert_manager.paraDomain, inVec=cov, destPos=grid.cellCenters())

        classes_grd = slopestabilitytools.assign_class01(input_model2_grd, resistivity_map)

        classesn_grd = slopestabilitytools.assign_classes(slopestabilitytools.normalize(input_model2_grd_array))

        labels_grd = slopestabilitytools.classes_labels.numeric2label(classesn_grd)

        experiment_results_grid = pd.DataFrame(data={'NAME': profil_name_column_grd,
                                                     'X': grid.cellCenters().array()[:, 0],
                                                     'Y': grid.cellCenters().array()[:, 1],
                                                     'Z': grid.cellCenters().array()[:, 2],
                                                     'INM': input_model2_grd_array,
                                                     'INMN': input_model2_grd_array_norm,
                                                     'RES': result_grd_array,
                                                     'RESN': result_grd_array_norm,
                                                     'SEN': cov_grd,
                                                     'CLASS': classes_grd,
                                                     'CLASSN': classesn_grd,
                                                     'LABELS': labels_grd})

        experiment_results_grid = experiment_results_grid[
            (experiment_results_grid.INMN > 0) & (experiment_results_grid.RESN > 0)]

        experiment_results_grid.to_csv(settings.settings['data_folder_grd'] + '/' + profile_name + '_grd.csv')

    return True
