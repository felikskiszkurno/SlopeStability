#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15.01.2021

@author: Feliks Kiszkurno
"""

import os
from .check_create_folder import check_create_folder


def create_folder_structure():

    is_success = True

    # Folder for figures
    folder_path = os.getcwd()+'/results/'
    is_success = check_create_folder(folder_path)
    folder_path = os.getcwd() + '/results/figures/'
    is_success = check_create_folder(folder_path)

    # Folder for results
    folder_path = os.getcwd()+'/results/'
    is_success = check_create_folder(folder_path)
    folder_path = os.getcwd() + '/results/results/'
    is_success = check_create_folder(folder_path)

    return is_success
