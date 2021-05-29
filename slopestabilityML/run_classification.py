#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19.01.2021

@author: Feliks Kiszkurno
"""
import settings
import slopestabilityML
import pandas as pd
import numpy as np
from scipy import interpolate

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import slopestabilitytools
import test_definitions


def run_classification(test_training, test_prediction, test_results, clf, clf_name, *, hyperparameters=False):

    accuracy_result = []
    accuracy_labels = []

    accuracy_result_training = []
    accuracy_labels_training = []

    depth_estim = []
    depth_true = []
    depth_estim_accuracy = []
    depth_estim_labels = []

    depth_estim_training = []
    depth_true_training = []
    depth_estim_accuracy_training = []
    depth_estim_labels_training = []

    slopestabilityML.classification_predict(test_training, test_results, clf, clf_name)

    slopestabilityML.classification_train()

    return result_class, accuracy_labels, accuracy_result, accuracy_labels_training, accuracy_result_training,\
           depth_estim, depth_true, depth_estim_accuracy, depth_estim_labels, depth_estim_training,\
           depth_true_training, depth_estim_accuracy_training, depth_estim_labels_training
