#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15.01.2021

@author: Feliks Kiszkurno
"""

import os
from pathlib import Path
from .check_create_folder import check_create_folder


def create_folder_structure():

    is_success = True

    # Folder for figures
    folder_path = Path(os.getcwd()+'/results/')
    is_success = check_create_folder(folder_path)
    folder_path = Path(os.getcwd() + '/results/figures/')
    is_success = check_create_folder(folder_path)
    folder_path = Path(os.getcwd() + '/results/figures/pdf')
    is_success = check_create_folder(folder_path)
    folder_path = Path(os.getcwd() + '/results/figures/eps')
    is_success = check_create_folder(folder_path)
    folder_path = Path(os.getcwd() + '/results/figures/png')
    is_success = check_create_folder(folder_path)
    folder_path = Path(os.getcwd() + '/results/figures/ML/pdf')
    is_success = check_create_folder(folder_path)
    folder_path = Path(os.getcwd() + '/results/figures/ML/eps')
    is_success = check_create_folder(folder_path)
    folder_path = Path(os.getcwd() + '/results/figures/ML/png')
    is_success = check_create_folder(folder_path)
    folder_path = Path(os.getcwd() + '/results/figures/ML/training/')
    is_success = check_create_folder(folder_path)
    folder_path = Path(os.getcwd() + '/results/figures/ML/training/pdf')
    is_success = check_create_folder(folder_path)
    folder_path = Path(os.getcwd() + '/results/figures/ML/training/eps')
    is_success = check_create_folder(folder_path)
    folder_path = Path(os.getcwd() + '/results/figures/ML/training/png')
    is_success = check_create_folder(folder_path)
    folder_path = Path(os.getcwd() + '/results/figures/ML/prediction/pdf')
    is_success = check_create_folder(folder_path)
    folder_path = Path(os.getcwd() + '/results/figures/ML/prediction/eps')
    is_success = check_create_folder(folder_path)
    folder_path = Path(os.getcwd() + '/results/figures/ML/prediction/png')
    is_success = check_create_folder(folder_path)

    # Folder for results
    folder_path = Path(os.getcwd()+'/results/')
    is_success = check_create_folder(folder_path)
    folder_path = Path(os.getcwd() + '/results/results/')
    is_success = check_create_folder(folder_path)

    return is_success
