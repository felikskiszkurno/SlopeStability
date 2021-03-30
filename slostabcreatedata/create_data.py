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


def create_data(test_name, test_config, max_depth):

    world_boundary_v = [-9 * max_depth, 0]  # [NW edge] relatively to the middle
    world_boundary_h = [9 * max_depth, -9 * max_depth]  # [SE edge]
    # world_boundary_v = [-500, 0]  # [right, left border] relatively to the middle
    # world_boundary_h = [500, -100]  # [top, bottom border]

    test_results = {}

    # INPUT MODEL - SUBSURFACE START #
    world = mt.createWorld(start=world_boundary_v, end=world_boundary_h,
                           layers=test_config['layers_pos'])  # ,
    # marker=np.linspace(1, tests_horizontal['layer_n']['hor_1'],
    #                  tests_horizontal['layer_n']['hor_1']))

    geometry = world  # +block

    fig_geometry, ax_geometry = plt.subplots(1)
    pg.show(geometry, ax=ax_geometry)
    ax_geometry = slopestabilitytools.set_labels(ax_geometry)
    ax_geometry.set_title('1 Geometry of the model')
    fig_geometry.tight_layout()
    slopestabilitytools.save_plot(fig_geometry, test_name, '_1_geometry')
    # fig_geometry.savefig('results/figures/png/' + test_name + '_1_geometry.png', bbox_inches="tight")
    #fig_geometry.savefig('results/figures/pdf/' + test_name + '_1_geometry.pdf', bbox_inches="tight")
    #fig_geometry.savefig('results/figures/eps/' + test_name + '_1_geometry.eps', bbox_inches="tight")

    measurement_scheme = ert.createERTData(
        elecs=np.linspace(start=-8 * max_depth, stop=8 * max_depth, num=8 * max_depth + 1),
        schemeName='dd')

    for electrode in measurement_scheme.sensors():
        geometry.createNode(electrode)
        geometry.createNode(electrode - [0, 0.1])  # What does it do?

    mesh = mt.createMesh(geometry, quality=34)  # , area=2)#

    resistivity_map = test_config['rho_values']  # [0]

    fig_model, ax_model = plt.subplots(1)
    pg.show(mesh, data=resistivity_map, label=pg.unit('res'), showMesh=True, ax=ax_model)
    ax_model = slopestabilitytools.set_labels(ax_model)
    ax_model.set_title('2 Mesh and resistivity distribution')
    fig_model.tight_layout()
    plot_name = '_2_meshdist.png'
    slopestabilitytools.save_plot(fig_model, test_name, '_2_meshdist')
    #fig_model.savefig('results/figures/png/' + test_name + '_2_meshdist.png', bbox_inches="tight")
    #fig_model.savefig('results/figures/pdf/' + test_name + '_2_meshdist.pdf', bbox_inches="tight")
    #fig_model.savefig('results/figures/eps/' + test_name + '_2_meshdist.eps', bbox_inches="tight")

    input_model = pg.solver.parseMapToCellArray(resistivity_map, mesh)  # rename to input_mesh

    # INPUT MODEL - SUBSURFACE MODEL END ###

    # SIMULATE ERT MEASUREMENT - START ###
    data = ert.simulate(mesh, scheme=measurement_scheme, res=resistivity_map, noiseLevel=1, noiseAbs=1e-6, seed=1337)
    data.remove(data['rhoa'] < 0)
    # SIMULATE ERT MEASUREMENT - END ###

    ert_manager = ert.ERTManager(sr=False, useBert=True, verbose=True, debug=False)

    # RUN INVERSION #
    k0 = pg.physics.ert.createGeometricFactors(data)
    model_inverted = ert_manager.invert(data=data, lam=20, paraDX=0.25, paraMaxCellSize=2, paraDepth=2 * max_depth,
                                        quality=34, zPower=0.4)

    result_full = ert_manager.inv.model

    # result_array = pg.utils.interperc(result_full, 5)
    # result_lim = result_full.array()
    # result_lim[np.where(result_array > max(resistivity_map[1]))] = float("NaN")
    # result_lim[np.where(result_array < min(resistivity_map[1]))] = float("NaN") # min(resistivity_map[1])
    # result_array = result_lim
    result_array = result_full.array()
    # result_array_norm = slopestabilitytools.normalize(result_array)

    fig_result, ax_result = plt.subplots(1)
    pg.show(ert_manager.paraDomain, result_full, label=pg.unit('res'), showMesh=True, ax=ax_result)
    ax_result = slopestabilitytools.set_labels(ax_result)
    ax_result.set_title('3 Result')
    fig_result.tight_layout()
    slopestabilitytools.save_plot(fig_result, test_name, '_3_result')
    # fig_result.savefig('results/figures/png/' + test_name + '_3_result.png', bbox_inches="tight")
    #fig_result.savefig('results/figures/pdf/' + test_name + '_3_result.pdf', bbox_inches="tight")
    #fig_result.savefig('results/figures/eps/' + test_name + '_3_result.eps', bbox_inches="tight")

    input_model2 = pg.interpolate(srcMesh=mesh, inVec=input_model, destPos=ert_manager.paraDomain.cellCenters())

    input_model2_array = input_model2.array()
    if settings.settings['clip'] is True:
        array_max = np.max(input_model2_array)
        array_min = np.min(input_model2_array)
        #input_model2_array = slostabcreatedata.clip_data(input_model2_array, array_max, array_min)
        result_array = slostabcreatedata.clip_data(result_array, array_max, array_min)
    input_model2_array_norm = slopestabilitytools.normalize(input_model2_array)



    fig_input, ax_input = plt.subplots(1)
    pg.show(ert_manager.paraDomain, input_model2, label=pg.unit('res'), showMesh=True, ax=ax_input)
    ax_input = slopestabilitytools.set_labels(ax_input)
    ax_input.set_title('4 Model on inv mesh')
    fig_input.tight_layout()
    slopestabilitytools.save_plot(fig_input, test_name, '_4_modelinv')
    #fig_input.savefig('results/figures/png/' + test_name + '_4_modelinv.png', bbox_inches="tight")
    #fig_input.savefig('results/figures/pdf/' + test_name + '_4_modelinv.pdf', bbox_inches="tight")
    #fig_input.savefig('results/figures/eps/' + test_name + '_4_modelinv.eps', bbox_inches="tight")

    #if not settings['norm_class']:

        # Create classes labels
    classes = []
    resistivity_values = []
    for pair in resistivity_map:
        resistivity_values.append(pair[1])
    # print(resistivity_values)

    # TODO: This has to be rewritten for more complicated cases
    for value in input_model2:
        # print(value)
        res_diff = np.abs(value * np.ones_like(resistivity_values) - resistivity_values)
        # print(res_diff)
        res_index = np.argmin(res_diff)
        # print(res_index)
        classes.append(res_index)

    #elif settings['norm_class']:

    classesn = slopestabilitytools.assign_classes(input_model2_array_norm)

    # Create sensitivity values
    jac = ert_manager.fop.jacobian()  #
    # Normalization only for visualization!

    # Coverage = cumulative sensitivity = all measurements
    cov = ert_manager.coverage()
    # pg.show(ert_manager.paraDomain, cov, label="Logarithm of cumulative sensitivity")

    rho_arr = []
    for entry in resistivity_map:
        rho_arr.append(entry[1])
    rho_arr = np.array(rho_arr)
    rho_max = np.max(rho_arr)
    rho_min = np.min(rho_arr)

    # TODO: this assumes only two resistivities, extend it to consider more
    result_array[np.where(result_array < rho_min)] = rho_min
    result_array[np.where(result_array > rho_max)] = rho_max

    result_array_norm = slopestabilitytools.normalize(result_array)

    experiment_results = pd.DataFrame(data={'X': ert_manager.paraDomain.cellCenters().array()[:, 0],
                                            'Y': ert_manager.paraDomain.cellCenters().array()[:, 1],
                                            'Z': ert_manager.paraDomain.cellCenters().array()[:, 2],
                                            'INM': input_model2_array,
                                            'INMN': input_model2_array_norm,
                                            'RES': result_array,
                                            'RESN': result_array_norm,
                                            'SEN': cov,
                                            'CLASS': classes,
                                            'CLASSN': classesn})

    # experiment_results.to_csv('results/results/'+test_name+'.csv')

    # test_results[test_name] = experiment_results

    experiment_results.to_csv(settings.settings['data_folder'] + '/' + test_name + '.csv')

    return experiment_results, rho_max, rho_min
