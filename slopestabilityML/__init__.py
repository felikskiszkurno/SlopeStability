#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 16.01.2021

@author: Feliks Kiszkurno
"""

from .combine_results import combine_results
from .plot_results import plot_results
from .run_every import run_all_tests
from .split_dataset import split_dataset
from .run_classification import run_classification
from .preprocess_data import preprocess_data
from .plot_class_res import plot_class_res
from .ask_committee import ask_committee
from .plot_class_overview import plot_class_overview
from .select_search_type import select_search_types
from .select_split_type import select_split_type
from .check_name import check_name
from .plot_depth_true_estim import plot_depth_true_estim
from .classification_train import classification_train
from .classification_predict import classification_predict


from .SVM.svm_run import svm_run
from .GBC.gbc_run import gbc_run
from .SGD.sgd_run import sgd_run
from .KNN.knn_run import knn_run
from .ADABOOST.adaboost_run import adaboost_run
from .RVM.rvm_run import rvm_run
from .MGC.max_grad_classi import max_grad_classi
from .MGC.mgc_run import mgc_run
