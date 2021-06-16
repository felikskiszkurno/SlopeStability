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


def invert_data(profile_name):

    # Load data
    ert_manager = ert.ERTManager(os.path.join(settings.settings['data_measurement'], profile_name+'.ohm'),
                                 useBert=True, verbose=True, debug=False)

    # RUN INVERSION
    #k0 = pg.physics.ert.createGeometricFactors(data)

    model_inverted = ert_manager.invert(lam=20, paraDX=0.25, paraMaxCellSize=2, zWeight=0.2,  # paraDepth=2 * max_depth,
                                        quality=34, zPower=0.4)

    return True
