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

from compute_anisotropy import analyze_image, get_anisotropy, trackmap

############################## USER PART ######################################

# pathes
imgpath     = os.getcwd()+os.sep+'MT_kinesin_blue.tif'
savedir     = '/home/quentin/Documents'# Path to where you want to save data/image outputs
framesave   = savedir+os.sep+'frames' 
stack       = True

# Detection parameters
feature_size    = 8
order_threshold = 0.5
R               = 15

# Tracking parameters
sR     = 30
memory = 5
filt   = 5

# display option
display_track = True




########################### functions call ####################################

# Performing detection
if not os.path.exists(savedir):
    os.makedirs(savedir)
e_vec, err_vec, cost_vec, theta_vec, phi, defect_char = analyze_image(imgpath, feature_size, R, order_threshold, stack=stack, savedir = savedir)


#e_vec, err_vec, cost_vec, theta_vec, phi, defect_char = get_anisotropy(imgpath, False, R, feature_size, bin_, fov, BoxSize, order_threshold, peak_threshold, stack=stack, savedir = savedir)

# Tracking
defect_char = tp.link(defect_char, sR, memory=memory)

# Creating the display images
if not os.path.exists(framesave):
    os.makedirs(framesave)
img = tf.imread(imgpath)
trackmap(img, defect_char, savedir=framesave, filt=filt, yes_traj=display_track)