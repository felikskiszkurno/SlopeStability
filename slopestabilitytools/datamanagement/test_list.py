#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 17.01.2021

@author: Feliks Kiszkurno
"""

import os

import settings

# TODO Add more flexibility in the way, that the path is handled


def test_list(extension):

    path = settings.settings['data_folder']
    file_list = os.listdir(path)

    test_names = []

    for file in file_list:

        test_names.append(file[:file.find(extension)])

    test_names = sorted(test_names)

    return test_names
