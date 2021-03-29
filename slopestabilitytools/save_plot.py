#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 25.03.2021

@author: Feliks Kiszkurno
"""

def save_plot(figure_handler, test_name, figure_name):
    figure_handler.savefig('results/figures/png/' + test_name + '_1_geometry'+'.png', bbox_inches="tight")

    # fig_geometry.savefig('results/figures/pdf/' + test_name + '_1_geometry.pdf', bbox_inches="tight")
    # fig_geometry.savefig('results/figures/eps/' + test_name + '_1_geometry.eps', bbox_inches="tight")

