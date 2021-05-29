#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 27.05.2021

@author: Feliks Kiszkurno
"""

import os
import settings
import slopestabilitytools


def recognize_batches():

    path = settings.settings['data_folder'] + 'prediction/'

    batch_list = []

    for batch_name in slopestabilitytools.datamanagement.find_objects(path):
        print(batch_name)
        batch_list.append(batch_name)

    return batch_list 
