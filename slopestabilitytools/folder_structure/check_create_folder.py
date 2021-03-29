#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15.01.2021

@author: Feliks Kiszkurno
"""

import os


def check_create_folder(folder_path):
    worked = True
    if os.path.isdir(folder_path):
        print("Folder for figures exists!")
    else:
        os.makedirs(folder_path)
        print("Created folder for figures!")

    return worked
