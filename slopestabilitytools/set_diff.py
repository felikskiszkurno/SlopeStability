#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 16.01.2021

@author: Feliks Kiszkurno
"""


def set_diff(input_list, input_set):

    diff_list = []

    for element in input_list:
        if element not in input_set:
            diff_list.append(element)

    return list(diff_list)
