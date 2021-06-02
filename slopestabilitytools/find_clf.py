#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30.05.2021

@author: Feliks Kiszkurno
"""

import os
import settings
import slopestabilitytools.datamanagement.test_list


def find_clf(abs_path_in=''):

    clf_names = slopestabilitytools.datamanagement.test_list('.sav',
                                                             abs_path=os.path.join(abs_path_in,
                                                                                   settings.settings['clf_folder']))

    return clf_names
