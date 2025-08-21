# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:04:20 2024

@author: Quentin

compute_anisotropy.py

These functions are used to compute anisotropy.


compute_angle_diagram
anisotropy_comparison
track_by_charge
reference_profile
crop_rotate_scalar

part of DeftPunk package, in processing subpackage
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import trackpy as tp
import scipy

plt.rcParams.update({'font.size': 16})
origin_file = os.path.abspath( os.path.dirname( __file__ ) )


def compute_angle_diagram(orientation, R, center=None, axis=0, plotthis = False, correction=True):
    """
    Compute the director field profile around a central point with radius R
    It uses interpolation.

    > azimuthal, director_f = compute_angle_diagram(orientation, R, center=None, axis=0, plotthis = False)

    Parameters
    ----------
    orientation : 2D array
        Director field.
    R : int
        Radius of contour around defect.
    center : size-2 array, optional
        Center of the contour. If None (default) the center is taken.
    axis : float, optional
        Orientation in rad of the defect tail. The default is 0.
    plotthis : Boolean, optional
        Do we plot the angular profile. The default is False.

    Returns
    -------
    phi : 1D array of floats
        Azimuthal angle.
    theta_unit : 1D array of float
        Director angle.

    """
    
    
    ### Define the contour where we will interpolate director field
    #Load the reference phi
    phi = np.load(origin_file+os.sep+'ref_epsilon'+os.sep+'orientationAzimuthal.npy')
    th_test = np.copy(orientation)
    s = orientation.shape
    x = np.arange(0, s[1])
    y = np.arange(0, s[0])
    if (center is None) or (center[0] is None):
        #assume it is in the middle
        center = [(s[1])/2, (s[0])/2]
        
    ### Interpolate the director on the contour points
    # go into nematic space
    tensorx = np.cos(2*orientation)
    tensory = np.sin(2*orientation)
    
    # create interpolation functions
    angle_interpx = scipy.interpolate.RegularGridInterpolator((y,x), tensorx, bounds_error=False)
    angle_interpy = scipy.interpolate.RegularGridInterpolator((y,x), tensory, bounds_error=False)
    
    # interpolate
    tx = np.ones(phi.shape)*np.nan
    ty = np.ones(phi.shape)*np.nan
    tensor_repx = angle_interpx((center[1]+R*np.sin(phi), center[0]+R*np.cos(phi)))
    tensor_repy = angle_interpy((center[1]+R*np.sin(phi), center[0]+R*np.cos(phi)))
    
    tensor_unitx = angle_interpx((center[1]+R*np.sin(phi+axis), center[0]+R*np.cos(phi+axis)))
    tensor_unity = angle_interpy((center[1]+R*np.sin(phi+axis), center[0]+R*np.cos(phi+axis)))
    
    
    tx = tensor_unitx
    ty = tensor_unity
    
    # go back to angle space
    theta_unit = ((np.arctan2(ty, tx)/2) - axis)%(np.pi)
    # have a continuously increasing vector
    theta_unit[np.logical_and(phi>3*np.pi/2, theta_unit<np.pi/4)] = theta_unit[np.logical_and(phi>3*np.pi/2, theta_unit<np.pi/4)]+np.pi
    theta_unit[np.logical_and(phi<np.pi/2, theta_unit>3*np.pi/4)] = theta_unit[np.logical_and(phi<np.pi/2, theta_unit>3*np.pi/4)]-np.pi
    
    theta_rep = np.arctan2(tensor_repy, tensor_repx)/2

        
    if plotthis:
        X, Y = np.meshgrid(x,y)
        plt.figure()
        plt.gca().invert_yaxis()
        plt.quiver(X,Y, np.cos(th_test), np.sin(th_test), angles='xy', pivot='mid', scale=50, width=.003, headaxislength=0, headlength=0, color='k')
        plt.quiver(center[0]+R*np.cos(phi), center[1]+R*np.sin(phi), R*np.cos(theta_rep), -R*np.sin(theta_rep), pivot='mid', scale=500, width=.003, headaxislength=0, headlength=0, color='r')
        plt.plot(center[0], center[1], 'o')
        plt.axis('scaled')
        
    if correction:
        if np.isnan(theta_unit[-1]):
            right_corr = 0
        else:
            right_corr = (theta_unit[-1]-np.pi)/2
        if np.isnan(theta_unit[0]):
            left_corr = 0
        else:
            left_corr = theta_unit[0]/2
        theta_unit = theta_unit - (right_corr + left_corr)/2
    
    if correction:
        theta_unit = theta_unit - (theta_unit[-1]-np.pi + theta_unit[0])/4
    
    return phi, theta_unit

def anisotropy_comparison(phi, theta, R=np.nan, path = 'DeftPunk'+os.sep+'processing'+os.sep+'ref_epsilon_shift'+os.sep):#r'.\ref_epsilon\\'


    if np.all(np.isnan(theta)):
        return [np.nan], [np.nan]
    if np.isnan(R):
        path = origin_file+os.sep+'ref_epsilon'+os.sep
        es = np.load(path + 'e_vec.npy')
        phi_ref = np.load(path + 'orientationAzimuthal.npy')
        costs = np.ones(es.shape)
    else:
        path = origin_file+os.sep+'ref_epsilon'+os.sep
        es = np.load(path + 'e_vec.npy')
        phi_ref = np.load(path + 'orientationAzimuthal.npy')
        xshift= np.load(path + os.sep + 'xshift.npy')
        costs = np.ones((len(es), len(xshift)))
    
    
    same = False
    if len(phi)==len(phi_ref):
        if np.all(phi==phi_ref):
            same = True
    
    if np.isnan(R):
        #safe = np.logical_and(phi>0.1, np.abs(phi-np.pi)>0.1, phi<6)
        for i in range(len(es)):
            th_ref = np.load(path+'orientationTheta_e%.2f.npy'%(es[i]))
            if not same:
                th_interp = scipy.interpolate.interp1d(phi_ref, th_ref)
                th_ref = th_interp(phi)
            #costs[i] = np.sqrt(np.sum(dphi*np.square(np.cos(2*th_ref)-np.cos(2*theta))+np.square(np.sin(2*th_ref)-np.sin(2*theta))))
            costs[i] = np.sqrt(np.nansum(np.square(np.arctan2(np.sin(2*(th_ref-theta)), np.cos(2*(th_ref-theta)))/2)))*2*np.pi/np.sum(np.logical_not(np.isnan(theta)))
            #costs[i] = np.sqrt(np.sum(np.square(th_ref[safe]-theta[safe])))

        return es, costs

    else:
        for i in range(len(es)):
            for j in range(len(xshift)):
                th_ref = np.load(path+'R%.0f'%(R)+os.sep+'Theta_e%.2f_xshift%.2f.npy'%(es[i], xshift[j]))
                if not same:
                    th_interp = scipy.interpolate.interp1d(phi_ref, th_ref)
                    th_ref = th_interp(phi)
                costs[i,j] = np.sqrt(np.nansum(np.square(np.arctan2(np.sin(2*(th_ref-theta)), np.cos(2*(th_ref-theta)))/2)))*2*np.pi/np.sum(np.logical_not(np.isnan(theta)))
        E, Shift = np.meshgrid(es, xshift)
        return E, Shift, costs
    
def track_by_charge(df, searchR, mem):
    ch = df['charge']
    minc = round(2*ch.min())/2
    maxc = round(2*ch.max())/2
    charray = np.arange(minc, maxc+0.5, 1/2)
    df['particle'] = np.nan
    for k in range(len(charray)):
        cond = np.abs(df['charge']-charray[k])<0.25
        if np.sum(cond)>2:
            linkeddf = tp.link_df(df[cond], search_range=searchR, memory=mem, pos_columns=['x', 'y'])
            if np.any(np.logical_not(np.isnan(df['particle']))):
                minpart = np.nanmax(df['particle'])
            else:
                minpart = 0
            df.loc[cond, 'particle']  = minpart + 1 + linkeddf['particle']
        
    return df            
                   
def reference_profile(e):
    """
    Returns the theoretical angular profile associated with the provided anisotropy.
    Reference profiles are stored in a folder.
    
    Parameters
    ----------
    e : float
        Splay-bend anisotropy.

    Returns
    -------
    ref_th : list of float
        Director angles associated with the provided play-bend anisotropy.
        The corresponding azimuthal angles are in ref_epsilon/orientationAzimuthal.npy

    """
    if np.isnan(e):
        phi = np.load('DeftPunk'+os.sep+'DeftPunk'+os.sep+'processing'+os.sep+'ref_epsilon'+os.sep+'orientationAzimuthal.npy')
        ref_th = np.ones(phi.shape)*np.nan
    else:
        if np.abs(e)<0.01:
            e = 0.
            ref_th = phi/2
        else:
            ref_th = np.load('DeftPunk'+os.sep+'DeftPunk'+os.sep+'processing'+os.sep+'ref_epsilon'+os.sep+'orientationTheta_e%.2f.npy'%(e))
    
    return ref_th        
        
def crop_rotate_scalar(field, axis, cropsize, xcenter=None, ycenter=None):

    sh = field.shape
    
    # the xcenter/ycenter are the indices of the center
    if xcenter is None:
        xcenter = round(sh[1]/2)
    else:
        xcenter = round(xcenter)
    
    if ycenter is None:
        ycenter = round(sh[0]/2)
    else:
        ycenter = round(ycenter)
        
    #center around middle point
    xc = xcenter-sh[1]/2
    yc = ycenter-sh[0]/2
    
    # plt.figure()
    # plt.imshow(field, cmap='gray')
    
    #rotate image and coordinates
    rot_field = scipy.ndimage.rotate(field, -axis*180/np.pi, reshape=True, cval=np.nan)
    xrotc = xc*np.cos(axis) - yc*np.sin(axis)
    yrotc = yc*np.cos(axis) + xc*np.sin(axis)
    # back into indices center
    sh = rot_field.shape
    xcenter = round(xrotc + sh[1]/2)
    ycenter = round(yrotc + sh[0]/2)
    
    # plt.figure()
    # plt.imshow(rot_field, cmap='gray')
    # plt.plot(xcenter, ycenter, 'o')
    
    
    #crop
    bigbox = cropsize
    # lx1 = xcenter - max(0, xcenter-bigbox)
    lx1 = min(bigbox, xcenter)
    lx2 = min(sh[1]-xcenter, bigbox)
    ly1 = min(bigbox, ycenter)
    ly2 = min(sh[0]-ycenter, bigbox)
    # ly1 = ycenter - max(0, ycenter-bigbox)
    # ly2 = min(sh[0]-ycenter, bigbox) + ycenter
    x1 = xcenter - min(lx1, lx2)
    x2 = xcenter + min(lx1, lx2)
    y1 = ycenter - min(ly1, ly2)
    y2 = ycenter + min(ly1, ly2)
    padx = bigbox-min(lx1, lx2)
    pady = bigbox-min(ly1, ly2)
    
    
    xcrop_ = np.arange(x1-xcenter,x2-xcenter)
    ycrop_ = np.arange(y1-ycenter,y2-ycenter)
    xcrop, ycrop = np.meshgrid(xcrop_, ycrop_)
    piece_defect = rot_field[y1:y2, x1:x2]
    
    # plt.figure()
    # plt.imshow(piece_defect, cmap='gray')
    
    #pad to reach required size
    piece_defect = np.pad(piece_defect, ((pady, pady), (padx, padx)), mode='constant', constant_values=np.nan)
    
    # plt.figure()
    # plt.imshow(piece_defect, cmap='gray')
    # print(piece_defect.shape)
    #rot_field = scipy.ndimage.rotate(piece_defect, -axis*180/np.pi, reshape=False, cval=np.nan)
    
    return xcrop, ycrop, piece_defect#rot_field
    
        
    
    
    
    
        