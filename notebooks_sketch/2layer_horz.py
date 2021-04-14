#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 10:29:00 2021

@author: felikskrno
"""

import numpy as np
from numpy import random
import matplotlib.pyplot as plt

import pygimli as pg
import pygimli.meshtools as mt
import pygimli.physics.ert as ert

random.seed(999)

number_of_tests = 1
rho_spread_factor = 1.5; rho_max = 150
layers_min = 1; layers_max = 3
world_boundary_v = [-10, 0]  # [right, left border] relatively to the middle
world_boundary_h = [50, -50]   # [top, bottom border]

test_names = {}
test_n_layers = {}
test_rho = {}
test_layers_pos = {}

for test_id in range(number_of_tests):
    
    test_names[test_id] = 'hor_{}'.format(str(test_id+1))
    test_n_layers[test_names[test_id]] =  np.random.randint(layers_min,layers_max)
    rho_temp = []; rho_used = []
    layer_pos_temp = []; layer_pos_used = []
    layer = 0
    
    # print(test_n_layers[test_names[test_id]]+2)
    
    while layer < test_n_layers[test_names[test_id]]:
        
        
                
        new_layer_pos = random.randint(world_boundary_v[0], world_boundary_v[1])
        
        if len(layer_pos_temp) == 0:
            
            layer_pos_temp.append(new_layer_pos)
            layer_pos_used.append(new_layer_pos)
            
        else:
            
            if new_layer_pos not in layer_pos_used:
                
                new_layer_pos = random.randint(world_boundary_v[0], world_boundary_v[1])
                layer_pos_temp.append(new_layer_pos)
                layer_pos_used.append(new_layer_pos)
                
            else: 
                
                while new_layer_pos in layer_pos_used:
                    
                    new_layer_pos = random.randint(world_boundary_v[0], world_boundary_v[1])
                    
                layer_pos_temp.append(new_layer_pos)
                layer_pos_used.append(new_layer_pos)
                
        layer += 1
    
    layer = 0
        
    while layer < test_n_layers[test_names[test_id]] + 1:
        
        new_rho = int(random.rand(1)[0]*rho_max)
        
        if len(rho_temp) == 0:
            
            rho_temp.append([layer+1, new_rho])
            rho_used.append(new_rho)
            
        else:
            
            if new_rho not in rho_used:
                
                new_rho = int(random.rand(1)[0]*rho_max)
                rho_temp.append([layer+1, new_rho])
                rho_used.append(new_rho)
            
            else: 
                
                while new_rho in rho_used:
                    
                    new_rho = int(random.rand(1)[0]*rho_max)
                
                rho_temp.append([layer+1, new_rho])
                rho_used.append(new_rho)
        
        layer += 1
        
    test_layers_pos[test_names[test_id]] = np.flip(np.sort(np.array(layer_pos_temp)))
    test_rho[test_names[test_id]] = rho_temp
    rho_temp = []; rho_used = []
    layer_pos = []; layer_pos_temp = []; layer_pos_used = []

tests_horizontal = {'names':test_names,
                    'layer_n':test_n_layers,
                    'rho_values':test_rho,
                    'layers_pos':test_layers_pos}   




world_boundary_v = [-200, 0]  # [right, left border] relatively to the middle
world_boundary_h = [200, -100]   # [top, bottom border]

test_results = {}
test_results_grid = {}
test_input = {}

test_name = 'hor_1'

print(test_name)
print(tests_horizontal['layer_n'][test_name])
print(tests_horizontal['layers_pos'][test_name])



### INPUT MODEL - SUBSURFACE START ###
world = mt.createWorld(start=world_boundary_v, end=world_boundary_h, layers=tests_horizontal['layers_pos'][test_name], marker=np.linspace(1,tests_horizontal['layer_n']['hor_1'], tests_horizontal['layer_n']['hor_1']))
#block = mt.createCircle(pos=[-5, -3.], radius=[4, 1], marker=4,
#                        boundaryMarker=10, area=0.1)
geometry = world #+block

measurement_scheme = ert.createERTData(elecs=np.linspace(start=-45, stop=45, num=91), schemeName='dd')
for electrode in measurement_scheme.sensors():
        
    geometry.createNode(electrode)
    geometry.createNode(electrode - [0, 0.1])  # What does it do?
        
mesh = mt.createMesh(geometry, quality=34)#, area=2)#
  
#print(tests_parameters['rho_values'][test_name])
resistivity_map = tests_horizontal['rho_values'][test_name]#[0]
resistivity_map[0] = [1, 50.0]
resistivity_map[1] = [2, 150.0]
#resistivity_map.append([3, 200])
#resistivity_map.append([4, 250])
#print(resistivity_map)
input_model = pg.solver.parseMapToCellArray(resistivity_map, mesh) # rename to input_mesh
#print(input_model)

### INPUT MODEL - SUBSURFACE MODEL END ###


#mesh_fig = pg.show(mesh, markers=True)
#mesh_fig_print = mesh_fig[0]
#print(mesh_fig)
#mesh_fig_print.saveas('data/hor_tests/Mesh_for_{}'.format(test_name), format='eps')
   


### SIMULATE ERT MEASUREMENT - START ###
mesh_pd = [] # add new mesh
data = ert.simulate(mesh, scheme=measurement_scheme, res=resistivity_map, noiseLevel=1, noiseAbs=1e-6, seed=1337)
data.remove(data['rhoa'] < 0)
#print('len data(r)')
#print(len(data['r']))
### SIMULATE ERT MEASUREMENT - END ###


ert_manager = ert.ERTManager(sr=False, useBert=True, verbose=True, debug=False)


### RUN INVERSION ###
k0 = pg.physics.ert.createGeometricFactors(data)
model_inverted = ert_manager.invert(data=data, lam=20, paraDX=0.25, paraMaxCellSize=5, paraDepth=10, quality=34.0, zPower=0.4)
result = ert_manager.inv.model
res_np = result.array()
test_results[test_name] = res_np  

    
#meshPD = pg.Mesh(ert_manager.paraDomain) # Save copy of para mesh for plotting later
#inversionDomain = pg.createGrid(x=np.linspace(start=-50, stop=50, num=50),
#                           y=-pg.cat([0], pg.utils.grange(0.5, 50, n=25)),
#                           marker=2)
#grid = pg.meshtools.appendTriangleBoundary(inversionDomain, marker=1,
#                                      xbound=50, ybound=50)
##model = ert.invert(data, mesh=grid, lam=20, verbose=True) # result is on grid, rename to mesh_resutl
#
#### Result on the grid ###
#model = ert_manager.invert(data, mesh=grid, lam=20, verbose=True, quality=15) # 0result is on grid, rename to mesh_resutl
#modelPD = ert_manager.paraModel(model)
#result_grid = ert_manager.inv.model
#res_np_grid = result_grid.array()
### End of Result ###
#test_results_grid[test_name] = res_np_grid  
   







input_model2 = pg.interpolate(srcMesh=mesh, inVec=input_model, destPos=ert_manager.paraDomain.cellCenters())
#input_model3 = pg.interpolate(srcMesh=mesh, inVec=input_model, destPos=grid)
fig, ax = plt.subplots(4)
fig.suptitle(test_name)
pg.show(mesh,input_model, ax=ax[0])
ax[0].set_title('Geometry of the model')

pg.show(ert_manager.paraDomain,input_model2, ax=ax[1])
ax[1].set_title("Model on the output mesh")


pg.show(ert_manager.paraDomain,result, ax=ax[2])
ax[2].set_title("Inverted")

pg.show(ert_manager.paraDomain,result-input_model2, ax=ax[3])
ax[3].set_title("Diff")

fig.savefig('test1.png')


fig_input, ax_input = plt.subplots(1)
pg.show(mesh,input_model, ax=ax_input)
ax_input.set_title('1 Geometry of the model')
fig_input.show()

fig_input2, ax_input2 = plt.subplots(1)
pg.show(ert_manager.paraDomain,input_model2, ax=ax_input2)
ax_input2.set_title("2 Model on the output mesh")
fig_input2.show()

fig_inverted, ax_inverted = plt.subplots(1)
pg.show(ert_manager.paraDomain,result, ax=ax_inverted, cMin=50, cMax=150)
ax_inverted.set_title("3 Inverted")
fig_inverted.show()

fig_diff, ax_diff = plt.subplots(1)
pg.show(ert_manager.paraDomain,result-input_model2, ax=ax_diff)
ax_diff.set_title("4 Diff")
fig_diff.show()

fig_diff.savefig('test2.png')

#pg.show(ert_manager.paraDomain,model)
#pg.show(ert_manager.paraDomain,mod)
#pg.show(ert_manager.paraDomain,model)

