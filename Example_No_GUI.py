# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 14:09:40 2025

@author: Quentin
"""

import numpy as np
import os
import trackpy as tp
import tifffile as tf
import sys
function_directory = os.getcwd()
sys.path.append(function_directory+os.sep+'functions'+os.sep)

from compute_anisotropy import get_anisotropy, trackmap

############################## USER PART ######################################

# pathes
imgpath = './MT_kinesin_blue.tif'
savedir = '../defect_analysis_documents/test_outputs'
framesave = savedir+os.sep+'frames'

# display option
display_track = True

# director field computation parameters
stack   = True  # is the image a stack?
sigma   = 8    # Size of window for director field computation
bin_    = 4     # Binning of the field wrt to image px

# Defect detection parameters
fov     = 2   # Size of the averaging gaussian window used on the field for nematic order parameter estimation
BoxSize = 8     # Size of box contour for charge computation
order_threshold = 0.5 # Threshold nematic order parameter under which we estimate a defect is present
peak_threshold  = 0.85 # Angle threshold for charge computation

# Anisotropy computation parameters
R = 40    # Size of intergation contour around defect to compute anisotropy

# Tracking parameters
sR     = 30
memory = 5
filt   = 5

########################### functions call ####################################

# Performing detection
if not os.path.exists(savedir):
    os.makedirs(savedir)
e_vec, err_vec, cost_vec, theta_vec, phi, defect_char = get_anisotropy(imgpath, False, R, sigma, bin_, fov, BoxSize, order_threshold, peak_threshold, stack=stack, savedir = savedir)

# Tracking
defect_char = tp.link(defect_char, sR, memory=memory)

# Creating the display images
if not os.path.exists(framesave):
    os.makedirs(framesave)
img = tf.imread(imgpath)
trackmap(img, defect_char, savedir=framesave, filt=filt, yes_traj=display_track)