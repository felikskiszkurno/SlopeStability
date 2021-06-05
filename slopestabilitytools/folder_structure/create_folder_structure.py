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


def create_folder_structure(batch_names=''):

    is_success = True

    # Folder for figures
    if batch_names is not '':
        for batch_name in batch_names:
            for file_format in settings.settings['plot_formats']:

                folder_path = os.path.join(settings.settings['figures_folder'], batch_name, file_format)
                is_success = check_create_folder(folder_path)

                folder_path = os.path.join(settings.settings['figures_folder'], batch_name, 'ML', file_format)
                is_success = check_create_folder(folder_path)

                folder_path = os.path.join(settings.settings['figures_folder'], batch_name, 'ML', 'prediction', file_format)
                is_success = check_create_folder(folder_path)

                folder_path = os.path.join(settings.settings['figures_folder'], batch_name, 'ML', 'feature_importance',
                                           file_format)
                is_success = check_create_folder(folder_path)
    else:
        for file_format in settings.settings['plot_formats']:
            folder_path = os.path.join(settings.settings['figures_folder'], file_format)
            is_success = check_create_folder(folder_path)

            folder_path = os.path.join(settings.settings['figures_folder'], 'ML', file_format)
            is_success = check_create_folder(folder_path)

            folder_path = os.path.join(settings.settings['figures_folder'], 'ML', 'prediction', file_format)
            is_success = check_create_folder(folder_path)

            folder_path = os.path.join(settings.settings['figures_folder'], 'ML', 'feature_importance',
                                       file_format)
            is_success = check_create_folder(folder_path)


    # There has to be a training folder in figures, that doesn't contain batch name
    for file_format in settings.settings['plot_formats']:

        folder_path = os.path.join(settings.settings['figures_folder'], 'ML', 'training', file_format)
        is_success = check_create_folder(folder_path)

        folder_path = os.path.join(settings.settings['figures_folder'], 'ML', 'feature_importance', file_format)
        is_success = check_create_folder(folder_path)

    # Folder for results
    folder_path = os.path.join(settings.settings['data_folder'])
    is_success = check_create_folder(folder_path)

    # Folder for classifiers
    folder_path = os.path.join(settings.settings['clf_folder'])
    is_success = check_create_folder(folder_path)

    folder_path = os.path.join(settings.settings['data_folder_grd'])
    is_success = check_create_folder(folder_path)

    return is_success
