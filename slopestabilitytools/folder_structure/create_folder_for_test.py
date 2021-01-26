#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15.01.2021

@author: Feliks Kiszkurno
"""

import os
from pathlib import Path
from .check_create_folder import check_create_folder


def create_folder_for_test(test_name):

    folder_path = Path(os.getcwd() / '/results/results/%'.format(test_name))
    is_success = check_create_folder(folder_path)

    return is_success
