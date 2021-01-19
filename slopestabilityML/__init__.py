#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 16.01.2021

@author: Feliks Kiszkurno
"""

from .combine_results import combine_results
from .plot_results import plot_results
from .run_all_tests import run_all_tests
from .split_dataset import split_dataset
from .run_classification import run_classification
from .preprocess_data import preprocess_data

from .SVM.svm_run import svm_run
from .GBC.gbc_run import gbc_run
from .SGD.sgd_run import sgd_run


