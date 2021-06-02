#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 25.03.2021

@author: Feliks Kiszkurno
"""

import os
import settings


def save_plot(figure_handler, test_name, figure_name, *, subfolder='', skip_fileformat=False, batch_name=''):  # keep in mind, that subfolder has to exist, it wont be created here

    '''if isinstance(subfolder, str) is True:
        subfolder = '/' + subfolder + '/'
    else:
        subfolder = '''''

    for file_format in settings.settings['plot_formats']:
        if skip_fileformat is True:  # What is this about?
            fig_name_ext = test_name + figure_name + '.' + file_format
            figure_handler.savefig(os.path.join(settings.settings['figures_folder'], batch_name,
                                                subfolder, fig_name_ext),
                                   bbox_inches="tight")
        else:
            fig_name_ext = test_name + figure_name + '.' + file_format
            figure_handler.savefig(os.path.join(settings.settings['figures_folder'], batch_name,
                                                subfolder, file_format, fig_name_ext),
                                   bbox_inches="tight")
