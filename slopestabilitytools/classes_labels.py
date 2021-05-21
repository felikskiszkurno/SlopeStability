#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 12.04.2021

@author: Feliks Kiszkurno
"""


def numeric2label(input_numeric):

    labels_translator = {0: 'Very Low',
                         1: 'Low',
                         2: 'Medium',
                         3: 'High',
                         4: 'Very High'}
    labels = [None] * len(input_numeric)
    find_all = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
    for key in labels_translator.keys():
        id_new = find_all(key, input_numeric)
        for idx in id_new:
            labels[idx] = labels_translator[key]

    return labels


def label2numeric(input_labels):

    labels_translator = {'Very Low': 0,
                         'Low': 1,
                         'Medium': 2,
                         'High': 3,
                         'Very High': 4}
    numeric = [None] * len(input_labels)
    find_all = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
    for key in labels_translator.keys():
        id_new = find_all(key, input_labels)
        for idx in id_new:
            numeric[idx] = labels_translator[key]

    return numeric