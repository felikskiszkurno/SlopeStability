#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 16.01.2021

@author: Feliks Kiszkurno
"""

from sklearn import svm
import slopestabilitytools
import random
import math

def svm_run(test_results):
    # https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/

    test_number = len(test_results.keys())
    test_prediction = random.choice(list(test_results.keys()),
                                    k=math.ceil(test_number*0.1))

    test_training = slopestabilitytools.set_diff(list(test_results.keys()), set(test_prediction))

    # Create classifier
    clf = svm.SVC(gamma=0.001, C=100)

    # Train classifier

    for test_name in test_training:

        # Prepare data
        data_set = test_results[test_name]
        X = data_set.drop('Z', '')

    # Predict with classfier

    # Plot

    return
