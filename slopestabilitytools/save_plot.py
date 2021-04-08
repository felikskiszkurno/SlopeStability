#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 25.03.2021

@author: Feliks Kiszkurno
"""

import os
from pathlib import Path

import settings


def save_plot(figure_handler, test_name, figure_name, *, subfolder=False, skip_fileformat=False):  # keep in mind, that subfolder has to exist, it wont be created here

    if isinstance(subfolder, str) is True:
        subfolder = '/' + subfolder + '/'
    else:
        subfolder = ''

    for file_format in settings.settings['plot_formats']:
        if skip_fileformat is True:
            figure_handler.savefig(Path(os.getcwd() + '/' + settings.settings[
                'figures_folder'] + subfolder + '/' + test_name + figure_name + '.' + file_format),
                                   bbox_inches="tight")
        else:
            figure_handler.savefig(Path(os.getcwd() + '/' + settings.settings[
                'figures_folder'] + subfolder + file_format + '/' + test_name + figure_name + '.' + file_format),
                                   bbox_inches="tight")