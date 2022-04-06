# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 10:43:25 2022

@author: INES
"""

import shutil
import os 

file_source = 'data/'
voltage_file_destination = 'data/voltage_borders/class_images/'
cond_file_detsination = 'data/conductivity_images/class_images/'

get_file = os.listdir(file_source)

for g in get_file:
     if '_voltage_border' in g:
         shutil.move(file_source+g, voltage_file_destination)
     if '_cond_img' in g:
         shutil.move(file_source+g, cond_file_detsination)



