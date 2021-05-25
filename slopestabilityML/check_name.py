#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 25.05.2021

@author: Feliks Kiszkurno
"""


def check_name(name_in):

    if name_in.find('_grd') != -1:
        name_out = name_in[0:name_in.find('_grd')]
    else:
        name_out = name_in

    return name_out, name_in
