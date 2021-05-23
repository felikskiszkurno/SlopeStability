#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23.05.2021

@author: Feliks Kiszkurno
"""

from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

import settings


def select_search_types(clf_base, hyperparameters):

    if settings.settings['optimize_ml_type'] is 'exhaustive':

        clf = GridSearchCV(clf_base, hyperparameters, n_jobs=-1, pre_dispatch='2*n_jobs')

    elif settings.settings['optimize_ml_type'] is 'halved':

        clf = HalvingGridSearchCV(clf_base, hyperparameters, n_jobs=-1, pre_dispatch='2*n_jobs')

    return clf
