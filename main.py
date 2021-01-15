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

for test_name in tests_horizontal['names'].values():
    slostabcreatedata.create_data()
world_boundary_v = [-200, 0]  # [right, left border] relatively to the middle
world_boundary_h = [200, -100]  # [top, bottom border]

test_results = {}
test_results_grid = {}
test_input = {}

test_name = 'hor_1'

#tests_horizontal['layers_pos'][test_name] = [-5]

# INPUT MODEL - SUBSURFACE START #
world = mt.createWorld(start=world_boundary_v, end=world_boundary_h, layers=tests_horizontal['layers_pos'][test_name])#,
                       #marker=np.linspace(1, tests_horizontal['layer_n']['hor_1'],
                        #                  tests_horizontal['layer_n']['hor_1']))

geometry = world  # +block

measurement_scheme = ert.createERTData(elecs=np.linspace(start=-45, stop=45, num=91), schemeName='dd')
for electrode in measurement_scheme.sensors():
    geometry.createNode(electrode)
    geometry.createNode(electrode - [0, 0.1])  # What does it do?

mesh = mt.createMesh(geometry, quality=34)  # , area=2)#

resistivity_map = tests_horizontal['rho_values'][test_name]  # [0]
#resistivity_map[0] = [1, 50.0]
#resistivity_map[1] = [2, 150.0]

input_model = pg.solver.parseMapToCellArray(resistivity_map, mesh)  # rename to input_mesh

# INPUT MODEL - SUBSURFACE MODEL END ###

# SIMULATE ERT MEASUREMENT - START ###
mesh_pd = []  # add new mesh
data = ert.simulate(mesh, scheme=measurement_scheme, res=resistivity_map, noiseLevel=1, noiseAbs=1e-6, seed=1337)
data.remove(data['rhoa'] < 0)
# SIMULATE ERT MEASUREMENT - END ###


ert_manager = ert.ERTManager(sr=False, useBert=True, verbose=True, debug=False)

# RUN INVERSION #
k0 = pg.physics.ert.createGeometricFactors(data)
model_inverted = ert_manager.invert(data=data, lam=20, paraDX=0.25, paraMaxCellSize=5, paraDepth=10, quality=34,
                                    zPower=0.4)
result = ert_manager.inv.model
result_array = result.array()

input_model2 = pg.interpolate(srcMesh=mesh, inVec=input_model, destPos=ert_manager.paraDomain.cellCenters())

input_model2_array = input_model2.array()

experiment_results = pd.DataFrame(data={'X': ert_manager.paraDomain.cellCenters().array()[:, 0],
                                        'Y': ert_manager.paraDomain.cellCenters().array()[:, 1],
                                        'Z': ert_manager.paraDomain.cellCenters().array()[:, 2],
                                        'INM': input_model2_array,
                                        'RES': result_array})

test_results[test_name] = experiment_results

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
