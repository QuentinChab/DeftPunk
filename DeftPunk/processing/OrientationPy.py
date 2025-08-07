# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 11:02:56 2023

@author: Quentin

You need skimage (scikit-image) version >= 0.18.x (I used 0.21.0)
IF USING SKIMAGE <0.20, ADD >>order='rc' as an argument of the feature.structuretensore()  function (lines 57 and 60 the 12/07/2024)

All I did was copy-paste Christoph Sommer from https://forum.image.sc/t/orientationj-or-similar-for-python/51767 
He used functions from skimage version <0.18.x and I replace with functions from more recent releases of skimage (used: 0.21.0).
I also added the option to plot with appropriate transformation
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
import skimage
import string
import math
import tifffile as tf

def dominant_direction(img, sigma):
    """OrientationsJ's dominant direction"""
    axx, axy, ayy = feature.structure_tensor(
        img.astype(np.float32), sigma=sigma, mode="reflect"
    )
    dom_ori = np.arctan2(2 * axy.mean(), (ayy.mean() - axx.mean())) / 2
    return np.rad2deg(dom_ori)

def orientation_analysis(img, sigma, binning, plotf=False, mode='pool'):    
    """Input
    img     = image or path to image
    sigma   = I think it refers to the size of the box on which you determine the direction
    binning = How much you downsize your arrow orientation display with respect to the image size
    plotf   = Do you want to plot the image with overlayed orientation arrays?
    &
    same as OrientationJ's output for
    - orientation
    - coherence
    - energy
    
    call
    orientation, coherence, ene, X, Y = orientation_analysis(img, sigma, binning, plotf=False)
    
    To display:
    plt.figure()
    plt.imshow(img) #to have the image. If you just want the vector field write plt.invert_yaxis() instead
    plt.quiver(X,Y,np.cos(orientation),np.sin(orientation), angles='xy', ....)
    
    The parameter angles='xy' is necessary because the default quiver indexing is different from the default image indexing

    """
    if skimage.__version__< "0.18":
        raise ImportError('Please update scikit image to a version >= 0.18')
    
    
    if isinstance(img, str):
        img = tf.imread(img)
        if img.ndim>2:
            img = np.mean(img[:,:,:2], axis=2)
    
    eps = 1e-20
    
    axx, axy, ayy = feature.structure_tensor(
        img.astype(np.float32), sigma=sigma, mode="reflect", order='rc'
    )
    A = feature.structure_tensor(
        img.astype(np.float32), sigma=sigma, mode="reflect", order='rc'
    )
    #plt.imshow(A)
    #print(axy)
    #print(ayy)
    l = feature.structure_tensor_eigenvalues(A)
    ori = np.arctan2(2 * axy, (ayy - axx)) / 2
    l1 = l[0]
    l2 = l[1]
    coh = ((l2 - l1) / (l2 + l1 + eps)) ** 2
    ene = np.sqrt(axx + ayy)
    ene /= ene.max()

    xbin = int(binning) 
    ybin = int(binning)
    N = img.shape
    
    # find the closest integer binning (greedy algorithm)
    # while N[0]%xbin != 0:
    #     xbin-=1
    # while N[1]%ybin != 0:
    #     ybin-=1
    
    if mode=='pool':
        # Pool instead of downsample
        thx = skimage.measure.block_reduce(np.cos(2*ori), (binning, binning), func=np.mean)
        thy = skimage.measure.block_reduce(np.sin(2*ori), (binning, binning), func=np.mean)
        orientation = np.arctan2(thy, thx)/2 - np.pi/2
        coherence = skimage.measure.block_reduce(coh, (binning, binning), func=np.mean)
        W, H = orientation.shape
        x_coords = np.arange(binning/2, H*binning, binning)
        y_coords = np.arange(binning/2, W*binning, binning)
        X, Y = np.meshgrid(x_coords, y_coords)  # Shape: (H//b, W//b)
    elif mode=='downsample':
        # In the image representation the axis are not the same as in plot representation 
        x1 = round((xbin + N[0]-xbin*math.floor(N[0]/xbin))/2) # To be like OrientationJ. 
        y1 = round((ybin + N[1]-ybin*math.floor(N[1]/ybin))/2)
        xpos    = np.arange(y1,N[1], ybin)
        ypos    = np.arange(x1,N[0], xbin)
        X,Y     = np.meshgrid(xpos,ypos)
        orientation = ori[x1::xbin,y1::ybin]-np.pi/2
        coherence = coh[x1::xbin,y1::ybin] #binned coh array
    else:
        print('chose a mode between pool and downsample')
        
    
    coherence[np.isnan(coherence)]=np.nanmin(coherence)
    
    if plotf:    
        plt.figure()
        plt.imshow(img, cmap='gray')
        plt.quiver(X, Y, np.cos(orientation), np.sin(orientation), angles='xy', scale=1/binning, width=1.5, headaxislength=0, headlength=0, pivot='mid', color='red', units='xy')
        # headaxislength = 0 and headlength=0 to remove arrows
        # units = 'xy' to get into absolute units (by default, relative to the image size)
        # pivot ='mid' so the point is at the middle of the body
        # scale and width control the rod length and width
        
    return orientation, coherence, ene, X, Y

def plot_vfield(X, Y, angle, plotimg = [], cfield='red', imgmap = 'gray'):
    f = plt.figure()
    
    if isinstance(plotimg, str):
        plotimg = tf.imread(plotimg)
    if len(plotimg)>0:
        plt.imshow(plotimg, cmap = imgmap)
        sc = len(X)/len(plotimg)
    else:
        plt.gca().invert_yaxis()
        sc = 1
    plt.quiver(X, Y, np.cos(angle), np.sin(angle), angles = 'xy', scale = sc, 
               headaxislength=0, headlength=0, pivot='mid', 
               color=cfield, units='xy')
    
    return f