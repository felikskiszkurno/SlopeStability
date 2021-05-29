#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 29.05.2021

@author: Feliks Kiszkurno
"""

import os


def find_objects(path, *, find_type='dir'):

    object_list = os.listdir(path)

    for item in object_list:
        if find_type == 'dir':
            print(path)
            print(item)
            if os.path.isfile(os.path.join(path, item)) is False:
                yield item
        elif find_type == 'file':
            if os.path.isfile(item) is True:
                yield item
        else:
            continue
