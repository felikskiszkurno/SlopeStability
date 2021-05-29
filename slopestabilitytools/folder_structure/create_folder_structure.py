#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15.01.2021

@author: Feliks Kiszkurno
"""

import os
from pathlib import Path

from .check_create_folder import check_create_folder
import settings


def create_folder_structure(batch_names=['batch1']):

    is_success = True

    # Folder for figures
    for batch_name in batch_names:
        for file_format in settings.settings['plot_formats']:

            folder_path = Path(os.getcwd() + '/' + settings.settings['figures_folder'] + '/' + batch_name + '/' + file_format)
            is_success = check_create_folder(folder_path)

            folder_path = Path(os.getcwd() + '/' + settings.settings['figures_folder'] + '/' + batch_name + '/' + 'ML/' + file_format)
            is_success = check_create_folder(folder_path)

            folder_path = Path(os.getcwd() + '/' + settings.settings['figures_folder'] + '/' + batch_name + '/' + 'ML/training/' + file_format)
            is_success = check_create_folder(folder_path)

            folder_path = Path(os.getcwd() + '/' + settings.settings['figures_folder'] + '/' + batch_name + '/' + 'ML/prediction/' + file_format)
            is_success = check_create_folder(folder_path)

    # Folder for results
    folder_path = Path(os.getcwd() + '/' + settings.settings['data_folder'])
    is_success = check_create_folder(folder_path)

    folder_path = Path(os.getcwd() + '/' + settings.settings['data_folder_grd'])
    is_success = check_create_folder(folder_path)

    return is_success
