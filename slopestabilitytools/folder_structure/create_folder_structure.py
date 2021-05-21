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


def create_folder_structure():

    is_success = True

    # Folder for figures
    for file_format in settings.settings['plot_formats']:

        folder_path = Path(os.getcwd() + '/' + settings.settings['figures_folder'] + file_format)
        is_success = check_create_folder(folder_path)

        folder_path = Path(os.getcwd() + '/' + settings.settings['figures_folder'] + 'ML/' + file_format)
        is_success = check_create_folder(folder_path)

        folder_path = Path(os.getcwd() + '/' + settings.settings['figures_folder'] + 'ML/training/' + file_format)
        is_success = check_create_folder(folder_path)

        folder_path = Path(os.getcwd() + '/' + settings.settings['figures_folder'] + 'ML/prediction/' + file_format)
        is_success = check_create_folder(folder_path)

    # Folder for results
    folder_path = Path(os.getcwd() + '/' + settings.settings['data_folder'])
    is_success = check_create_folder(folder_path)

    folder_path = Path(os.getcwd() + '/' + settings.settings['data_folder_grd'])
    is_success = check_create_folder(folder_path)

    return is_success
