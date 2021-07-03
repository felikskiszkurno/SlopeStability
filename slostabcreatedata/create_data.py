#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15.01.2021

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


def create_data(test_name, test_config, max_depth, *, lambda_param=20, z_weight=0.6):
    world_boundary_v = [-200, 0]  # [NW edge] relatively to the middle
    world_boundary_h = [200, -100]  # [SE edge]
    #world_boundary_v = [-1000, 0]  # [right, left border] relatively to the middle
    #world_boundary_h = [1000, -100]  # [top, bottom border]

    test_results = {}

    # INPUT MODEL - SUBSURFACE START #
    world = mt.createWorld(start=world_boundary_v, end=world_boundary_h,
                           layers=test_config['layers_pos'])  # ,
    # marker=np.linspace(1, tests_parameters['layer_n']['hor_1'],
    #                  tests_parameters['layer_n']['hor_1']))

    geometry = world  # +block

    fig_geometry, ax_geometry = plt.subplots(1)
    pg.show(geometry, ax=ax_geometry)
    ax_geometry = slopestabilitytools.set_labels(ax_geometry)
    ax_geometry.set_title('1 Geometry of the model; lam:{}, zw:{}'.format(lambda_param, z_weight))
    fig_geometry.tight_layout()
    slopestabilitytools.save_plot(fig_geometry, test_name, '_1_geometry')

    measurement_scheme = ert.createERTData(
        elecs=np.linspace(start=-32, stop=32, num=44),
        #elecs=np.linspace(start=-50, stop=50, num=100),
        schemeName='dd')

    for electrode in measurement_scheme.sensors():
        geometry.createNode(electrode)
        geometry.createNode(electrode - [0, 0.1])  # What does it do?

    mesh = mt.createMesh(geometry, quality=34, area=2)#

    resistivity_map = test_config['rho_values']  # [0]

    fig_model, ax_model = plt.subplots(1)
    pg.show(mesh, data=resistivity_map, label=pg.unit('res'), showMesh=True, ax=ax_model)
    ax_model = slopestabilitytools.set_labels(ax_model)
    ax_model.set_title('2 Mesh and resistivity distribution; lam:{}, zw:{}'.format(lambda_param, z_weight))
    fig_model.tight_layout()
    plot_name = '_2_meshdist.png'
    slopestabilitytools.save_plot(fig_model, test_name, '_2_meshdist')

    input_model = pg.solver.parseMapToCellArray(resistivity_map, mesh)  # rename to input_mesh

    # INPUT MODEL - SUBSURFACE MODEL END ###

    # SIMULATE ERT MEASUREMENT - START ###
    data = ert.simulate(mesh, scheme=measurement_scheme, res=resistivity_map, noiseLevel=1, noiseAbs=1e-6, seed=1337)
    data.remove(data['rhoa'] <= 0)
    # SIMULATE ERT MEASUREMENT - END ###

    ert_manager = ert.ERTManager(sr=False, useBert=True)# , verbose=True, debug=False)

    # RUN INVERSION #
    k0 = pg.physics.ert.createGeometricFactors(data)
    #inversion_Domain = mt.createMesh(mt.createRectangle([-35, 0], [35, -25], quality=34, area=1))
    #inversion_mesh = pg.meshtools.appendTriangleBoundary(inversion_Domain, marker=0, xbound=30, ybound=30)
    model_inverted = ert_manager.invert(data=data, lam=lambda_param, paraDX=0.25, paraMaxCellSize=2, zWeight=z_weight,# paraDepth=15,
                                        quality=34, zPower=0.4)

    result_full = ert_manager.inv.model

    result_array = result_full.array()

    limits = pg.utils.interperc(ert_manager.inv.model, trimval=10.0)
    result_array[result_array <= limits[0]] = limits[0]
    result_array[result_array >= limits[1]] = limits[1]

    fig_result, ax_result = plt.subplots(1)
    pg.show(ert_manager.paraDomain, result_full, label=pg.unit('res'), showMesh=True, ax=ax_result)
    ax_result = slopestabilitytools.set_labels(ax_result)
    ax_result.set_title('3 Result')
    fig_result.tight_layout()
    slopestabilitytools.save_plot(fig_result, test_name, '_3_result')

    input_model2 = pg.interpolate(srcMesh=mesh, inVec=input_model, destPos=ert_manager.paraDomain.cellCenters())

    input_model2_array = input_model2.array()
    if settings.settings['clip'] is True:
        array_max = np.max(input_model2_array)
        array_min = np.min(input_model2_array)
        result_array = slostabcreatedata.clip_data(result_array, array_max, array_min)
    input_model2_array_norm = np.log10(input_model2_array)
    # input_model2_array_norm = slopestabilitytools.normalize(input_model2_array)

    fig_input, ax_input = plt.subplots(1)
    pg.show(ert_manager.paraDomain, input_model2, label=pg.unit('res'), showMesh=True, ax=ax_input)
    ax_input = slopestabilitytools.set_labels(ax_input)
    ax_input.set_title('4 Model on inv mesh; lam:{}, zw:{}'.format(lambda_param, z_weight))
    fig_input.tight_layout()
    slopestabilitytools.save_plot(fig_input, test_name, '_4_modelinv')

    # Create classes labels
    '''
    classes = []
    resistivity_values = []
    for pair in resistivity_map:
        resistivity_values.append(pair[1])

    for value in input_model2:
        res_diff = np.abs(value * np.ones_like(resistivity_values) - resistivity_values)
        res_index = np.argmin(res_diff)
        classes.append(res_index)
    '''
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


    result_array_norm = np.log10(result_array)
    # result_array_norm = slopestabilitytools.normalize(result_array)

    labels = slopestabilitytools.classes_labels.numeric2label(classesn)
    '''
    labels_translator = {0: 'Very Low',
                         1: 'Low',
                         2: 'Medium',
                         3: 'High',
                         4: 'Very High'}
    labels = [None] * len(classesn)
    find_all = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
    for key in labels_translator.keys():
        id_new = find_all(key, classesn)
        for idx in id_new:
            labels[idx] = labels_translator[key]
    '''

    test_name_column = [test_name] * len(input_model2_array)

    # Results on unstructured grid
    experiment_results = pd.DataFrame(data={'NAME': test_name_column,
                                            'X': ert_manager.paraDomain.cellCenters().array()[:, 0],
                                            'Y': ert_manager.paraDomain.cellCenters().array()[:, 1],
                                            'Z': ert_manager.paraDomain.cellCenters().array()[:, 2],
                                            'INM': input_model2_array,
                                            'INMN': input_model2_array_norm,
                                            'RES': result_array,
                                            'RESN': result_array_norm,
                                            'SEN': cov,
                                            'CLASS': classes,
                                            'CLASSN': classesn,
                                            'LABELS': labels})
    experiment_results = experiment_results[(experiment_results.INMN > 0) & (experiment_results.RESN > 0)]

    experiment_results.to_csv(settings.settings['data_folder'] + '/' + test_name + '.csv')

    plot_title = '_in_inv_diff'
    if settings.settings['norm_class'] is True:
        slopestabilitytools.plot_and_save_pg(test_name, plot_title, ert_manager,
                                             input_model2, result_full, classesn)
        slopestabilitytools.plot_class_inv(classesn, ert_manager, test_name, plot_title)
    else:
        slopestabilitytools.plot_and_save_pg(test_name, plot_title, ert_manager,
                                             input_model2, result_full, classes)
        slopestabilitytools.plot_class_inv(classes, ert_manager, test_name, plot_title)

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

        input_model2_grd = pg.interpolate(srcMesh=ert_manager.paraDomain, inVec=input_model2, destPos=grid.cellCenters())

        input_model2_grd_array = input_model2_grd.array()

        test_name_column_grd = [test_name] * len(input_model2_grd_array)

        input_model2_grd_array_norm = np.log10(input_model2_grd)

        result_grd = pg.interpolate(srcMesh=ert_manager.paraDomain, inVec=result_full, destPos=grid.cellCenters())

        result_grd_array = result_grd.array()

        result_grd_array_norm = np.log10(result_grd_array)

        cov_grd = pg.interpolate(srcMesh=ert_manager.paraDomain, inVec=cov, destPos=grid.cellCenters())

        classes_grd = slopestabilitytools.assign_class01(input_model2_grd, resistivity_map)

        classesn_grd = slopestabilitytools.assign_classes(slopestabilitytools.normalize(input_model2_grd_array))

        labels_grd = slopestabilitytools.classes_labels.numeric2label(classesn_grd)

        experiment_results_grid = pd.DataFrame(data={'NAME': test_name_column_grd,
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

        experiment_results_grid = experiment_results_grid[(experiment_results_grid.INMN > 0) & (experiment_results_grid.RESN > 0)]

        experiment_results_grid.to_csv(settings.settings['data_folder_grd'] + '/' + test_name + '_grd.csv')

    return experiment_results, experiment_results_grid, rho_max, rho_min
