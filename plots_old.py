#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 16.01.2021

@author: Feliks Kiszkurno
"""


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