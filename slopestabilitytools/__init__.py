#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15.01.2021

@author: Feliks Kiszkurno
"""

""" This module is created to contain all tools, that are not directly related to inversion or processing
List:
model_params: generates parameters used to develop models
directory_structure: create directory structure to contain figures and other files
"""

from .model_params import model_params
from .plot_and_save import plot_and_save
from .set_labels import set_labels
from .set_diff import set_diff

from .folder_structure.create_folder_structure import create_folder_structure
from .folder_structure.create_folder_for_test import create_folder_for_test
from .folder_structure.check_create_folder import check_create_folder
