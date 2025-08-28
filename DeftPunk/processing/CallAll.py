# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:04:20 2024

@author: Quentin

CallAll.py

This module contains 4 functions that sequentially call functions performing
the defect detection.

- one_defect_anisotropy (compute anisotropy of one defect)
- get_anisotropy (performs director field computation, defect detection and anisotropy computation)
- analyze_image (wraps up get_anisotropy with less input parameters)
- anisotropy_on_directory (perform detection on each image of a selected folder)

Part of DeftPunk package
"""

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np
import os
import pandas as pd
from scipy.spatial.distance import cdist
import trackpy as tp
from math import floor
from DeftPunk.GUI_utils import load_image
from .OrientationPy import orientation_analysis
from .detect_defects import defect_detection
from .compute_anisotropy import compute_angle_diagram, anisotropy_comparison, track_by_charge, reference_profile

plt.rcParams.update({'font.size': 16})



def one_defect_anisotropy(field, R, xc=None, yc=None, axis = 0, err = 0.05, plotit=False):
    """
    From a director field and the location of a +1/2 defect, compute the anisotropy
    associated with the defect.

    emin, err_e, costmin, th_min = one_defect_anisotropy(field, R)

    Parameters
    ----------
    field : 2D array 
        Director field of with a +1/2 defect.
        If the center coordinates are nor provided the defect is assumed centered.
    R : float, optional
        Radius of detection for angular profile in director field unit.
    xc : float, optional
        x-coordinate of the defect in term of vector field.
        Default  is None, and then the center of the field is assumed.
    yc : float, optional
        y-coordinate of the defect in term of vector field.
        Default  is None, and then the center of the field is assumed.
    axis : float, otpional
        Axis of the tail of the defect with respect to x-direction, in rad.
        Default is 0.
    err : float, optional
        Used to estimate an error. The default is 0.05.
    plotit : bool, optional
        If True, plot the field, defect, contour on which the angular profile
        is computed. Also plot the angular profile.
        Defautl is False.

    Returns
    -------
    emin : float
        Output anisotropy.
    err_e : float
        Estimated error on anisotropy.
    costmin : float
        Value of the cost function for output anisotropy.
    th_min : 1D array
        orientation field at the detection radius. The corresponding azimuthal
        angle is stored

    """
    ## find the best fit anisotropy
    # get the director angle as a function of the azimutha angle
    phi_cycle, th_min   = compute_angle_diagram(field, R, center=[xc, yc], axis=axis, plotthis=plotit)
    # get the cost associated to each possible anisotropy
    es, costs           = anisotropy_comparison(phi_cycle, th_min)
    # index, anisotropy and cost for best fit anisotropy
    imin                = np.argmin(costs)
    emin                = es[imin]
    costmin             = costs[imin]
    
    ## compute error
    # The minimal cost is costmin. We probe the 2 anisotropies corresponding to
    # a cost of (1+err)*costmin. They will define the error.
    # by default err=0.05 -> cost = 1.05*costmin
    
    
    err_level = costs[imin]*(1+err) # error level to detect
    # lower bound
    if imin==0:
        ierr1 = 0
    else: 
        ierr1 = np.argmin(np.abs(costs[:imin]-err_level))
    
    # upper bound
    if imin ==len(costs):
        ierr2 = imin
    else:
        ierr2 = imin + np.argmin(np.abs(costs[imin:]-err_level))
    
    err_e = (es[ierr2] - es[ierr1])/2
    
    ## Display
    if plotit:
        
        plt.figure()
        
        plt.subplot(1,2,1)
        plt.plot(es, costs,'+')
        yl = plt.ylim()
        plt.plot([emin-err_e, emin-err_e], yl, 'k--')
        plt.plot([emin+err_e, emin+err_e], yl, 'k--')
        plt.xlabel('Anisotropy []')
        plt.ylabel('Cost [rad]')
        plt.tight_layout()
        
        plt.subplot(1,2,2)
        plt.plot(phi_cycle, th_min, 'o', label='Measure')
        fpath = os.path.abspath(__file__)
        th_ref = np.load(os.sep.join(fpath.split(os.sep)[:-1])+os.sep+'ref_epsilon'+os.sep+'orientationTheta_e%.2f.npy'%(emin))
        plt.plot(phi_cycle, th_ref, '--', label='Reference')
        plt.xlabel('Azimuthal angle [rad]')
        plt.ylabel('Director angle [rad]')
        plt.title(r'e=%.2f'%(emin))
        plt.legend()
        plt.tight_layout()
    
    
    return emin, err_e, costmin, th_min
   
def get_anisotropy(imgpath, R=np.nan, sigma=25, bin_=4, fov=2, BoxSize=6, order_threshold=0.25, peak_threshold=0.75, prescribed_field=None, plotit=False, stack=False, savedir = None, give_field=False):
    """
    From an image/path to an image or stack, compute the director field, detect the 
    defects and the anisotropy for +1/2 images
    There are 6 detection parameters, while we offer only 3 in the interfaces.
    (In inteface, some parameters are coupled with others)
    
    copy-paste:
    e_vec, err_vec, cost_vec, theta_vec, phi, defect_char = get_anisotropy(imgpath, R, sigma, bin_, fov, BoxSize, order_threshold, peak_threshold, plotit=False, stack=False, savedir = None)

    It is organize as such:
        - Handle image output
        - Treat single image input with
            1. Compute director field
            2. Detection of defects
            3. Computation of anisotropy
            4. Distance to nearest defect
        - Treat stack by inputing each frame into get_anisotropy
    
    Parameters
    ----------
    imgpath : string
        Path to the image.
    R : number, optional
        Radius of detection for anisotropy computation. 
        The default value is a function of the image size
    sigma : int, optional
        Averaging window for field computation. The default is 25.
    bin_ : int, optional
        binning of the field wrt image size. The default is 4.
    fov : int, optional
        Averaging window (on the field) for defect detection. The default is 2.
    BoxSize : int, optional
        Box size for defect charge computation. The default is 6.
    order_threshold : float, optional
        Threshold of order parameter to locate defect. The default is 0.25.
    peak_threshold : float, optional
        Threshold on angle jump to find charge. The default is 0.75.
    prescribed_field : 2D array or None
        Director field input by the user.
        If None (Default), the director field is computed.
    plotit : boolean
        If True, computation steps are displayed (mostly for debugging).
        Default is False.
    stack : boolean
        Is the image a stack? Only useful if imgpath is an image array.
        Default is False
    savedir : string or None
        Path where data will be saved (only if plotit is True).
        If None (default) it is not saved.
    give_field : bool, optional
        If True, return also the vector field and arrow positions.
        Default is False.

    Returns
    -------
    e_vec : 1D numpy array
        Array of anisotropies for all detected defects OR for the average
        defect if avergae=True.
    err_vec : 1D numpy array
        corresponding error.
    cost_vec : 1D numpy array
        Corresponding value of cost function.
    th_vec : list
        Corresponding angular profiles.
    phi : list
        Azimuthal angle corresponding to th_vec. It's actually always the same.
    defect_char : pandas DataFrame
        DataFrame containing the 
    orientation : 2D numpy array
        Angle of the director field. Only returned if give_field=True.
    pos : list of 2 1D arrays
        Corresponding arrow positions [x, y]. Only returned if give_field=True.

    """
    ####### Handle input: image or image path? ##############################
    # The input is either an image or a path to an image 
    if isinstance(imgpath, str):
        img, stack, _ = load_image(imgpath)
    else:
        img = imgpath
        
    
    ########### Detection for one frame #######################################
    ## compute director field, locate defects and compute anisotropy ##########
    if not stack: 
        ## If the director field is not an input, compute it        
        if prescribed_field is None:
            orientation, coherency, energy, x, y = orientation_analysis(img, sigma, bin_, plotit)        
        else:
            orientation = prescribed_field
            print(prescribed_field)
            coherency = np.ones(prescribed_field.shape)
            sh = orientation.shape
            x_ = np.arange(sh[1])
            y_ = np.arange(sh[0])
            x, y = np.meshgrid(x_, y_)     
        
        ## Peform detection of defects
        Qloc, boxes, chargeb, defect_axis, centroidsN = defect_detection(orientation, coherency, fov, BoxSize, order_threshold, peak_threshold, plotall=plotit, method='weighted')
    
        # convert 
        if not (img is None):
            # convert centroids from director field coordinates to image coordinates (multiply by bin_)
            N = img.shape
            x1 = round((bin_ + N[0]-bin_*floor(N[0]/bin_))/2) # To be like OrientationJ. 
            y1 = round((bin_ + N[1]-bin_*floor(N[1]/bin_))/2)
            img_centroids = np.empty(centroidsN.shape)
            img_centroids[:,0] = centroidsN[:,0]*bin_+x1
            img_centroids[:,1] = centroidsN[:,1]*bin_+y1
            

        else:
            img = np.ones(prescribed_field.shape)*np.nan
        
        ## Anisotropy computation 
        # Azimuthal coordinates
        fpath = os.path.abspath(__file__)
        phi = np.load(os.sep.join(fpath.split(os.sep)[:-1])+os.sep+'ref_epsilon'+os.sep+'orientationAzimuthal.npy')
        
        # will contain the anisotropy-related quantities
        e_vec       = [] # Anisotropy
        err_vec     = [] # error on anisotropy
        cost_vec    = [] # cost function for best-fit anisotropy
        theta_vec   = [] # angular profile (director angle)
        fields      = []
        
        for i in range(len(chargeb)):
            if np.abs(chargeb[i]-0.5)<0.1:
                # compute the anisotropy, error, cost and angular profile of a function
                e_vec_i, err_vec_i, cost_vec_i, th = one_defect_anisotropy(orientation, R, xc=centroidsN[i,1], yc=centroidsN[i,0], axis=defect_axis[i], plotit=plotit)
                e_vec.append(e_vec_i)
                err_vec.append(err_vec_i)
                cost_vec.append(cost_vec_i)
                theta_vec.append(th)
                fields.append(orientation)

        
        if plotit:
            indent = 0
            
            frange = plt.figure()
            plt.imshow(img, cmap='gray')
            plt.quiver(x,y,np.cos(orientation),np.sin(orientation),angles='xy',pivot='mid',headaxislength=0,headlength=0,scale=50)
            
            fmap, ax = plt.subplots()
            plt.imshow(img, cmap='gray')
            mycmap = 'PiYG'
            #☺colorm = cm.get_cmap('OrRd')
            colorm = cm.get_cmap(mycmap)
            for i in range(len(chargeb)):
                if chargeb[i]==0.5:
                    plt.figure(frange)
                    plt.plot(img_centroids[i,1],img_centroids[i,0],'o')
                    plt.plot(img_centroids[i,1]+R*bin_*np.cos(phi),img_centroids[i,0]+R*bin_*np.sin(phi), 'r')
                    
                    c = colorm(e_vec[indent]+0.5)
                    plt.figure(fmap)
                    plt.quiver(img_centroids[i,1], img_centroids[i,0], np.cos(defect_axis[i]), np.sin(defect_axis[i]), angles='xy', color=c)
                    plt.annotate('%.2f'%(e_vec[indent]), (img_centroids[i,1]+bin_, img_centroids[i,0]+bin_), color = c, fontsize='small')
                    indent += 1
    
                elif chargeb[i]==-0.5:
                    plt.quiver(img_centroids[i,1], img_centroids[i,0], np.cos(defect_axis[i]), np.sin(defect_axis[i]), angles='xy', color='b')
                    plt.quiver(img_centroids[i,1], img_centroids[i,0], np.cos(defect_axis[i]+2*np.pi/3), np.sin(defect_axis[i]+2*np.pi/3), angles='xy', color='b')
                    plt.quiver(img_centroids[i,1], img_centroids[i,0], np.cos(defect_axis[i]-2*np.pi/3), np.sin(defect_axis[i]-2*np.pi/3), angles='xy', color='b')
                else:
                    plt.plot(img_centroids[i,1], img_centroids[i,0], 'ko')
            plt.colorbar(cm.ScalarMappable(norm=Normalize(-0.5, 0.5), cmap=mycmap), label='Anisotropy []', ax=ax)
        
        # in some situations theta_vec cannot be empty
        if len(e_vec)==0:
            theta_vec = [np.ones(phi.shape)*np.nan]
        
        ## Compute distance to nearest defect
        ndef = len(chargeb)
        closest_neighbor = np.ones(ndef)*np.nan
        centroids_stack = [centroidsN[:,0]*bin_, centroidsN[:,1]*bin_]
        
        if ndef>2: #if more than 2 defects on the frame
            coord = np.array(centroids_stack).T
            dist_mat = cdist(coord, coord) #distance matrix for all defects in the frame
            for j in range(len(dist_mat)):
                if j==0:
                    closest_neighbor[j] = np.min(dist_mat[0, 1:])
                elif j==len(dist_mat)-1:
                    closest_neighbor[j] = np.min(dist_mat[j, :-1])
                else:
                    closest_neighbor[j] = min(np.min(dist_mat[j, :j]), np.min(dist_mat[j,j+1:])) 
        elif ndef==2:
            coord = np.array(centroids_stack).T
            dist_mat = cdist(coord, coord)
            closest_neighbor[0] = dist_mat[0][1]
            closest_neighbor[1] = dist_mat[0][1]
        elif ndef==1:
            closest_neighbor = np.nan
            
        # prepare output table (DataFrame) with defect detection information  
        defect_char = pd.DataFrame()
        defect_char['charge'] = chargeb
        defect_char['axis'] = defect_axis
        defect_char['x'] = centroidsN[:,1]*bin_
        defect_char['y'] = centroidsN[:,0]*bin_
        defect_char['Anisotropy'] = np.nan
        defect_char['Error'] = np.nan
        incr = 0 # count of +1/2 defects, for e_vec
        for di in range(len(defect_char)):
            if chargeb[di]==0.5: 
                # fill the anisotorpy information only for +1/2 defects
                defect_char.loc[di, 'Anisotropy'] = e_vec[incr]
                defect_char.loc[di, 'Error'] = err_vec[incr]
                incr +=1
        defect_char['MinDist'] = closest_neighbor
        ####
        
        defect_char = defect_char[np.abs(defect_char['charge'])>0.2]
        defect_char = defect_char.reset_index(drop=True)
        if give_field:
            return e_vec, err_vec, cost_vec, theta_vec, phi, defect_char, orientation, [x,y]
        else:
            return e_vec, err_vec, cost_vec, theta_vec, phi, defect_char


    ######## Now is the image is a stack ######################################
    ## the function will call itself in a a loop, frame by frame
    else: # if the image is a stack
        
        # initialize table and lists
        # Linear coordinates (lists have one dimension)
        defectdf = pd.DataFrame()
        xdf      = [] # x coord
        ydf      = [] # y coord
        tdf      = [] # frame
        edf      = [] # anisotropy
        errdf    = [] # error on anisotropy
        chargedf = [] # defect charge
        axisdf   = [] # defect orientation
        minddf   = [] # Distance to closest defect
        
        # To those lists, we will append the list of anisotropies for each frame (list of list) 
        # thesee ones will be returned
        e_stack         = []
        err_stack       = []
        cost_stack      = []
        theta_stack     = []
        phi_stack       = []
        centroids_stack = []
        axis_stack      = []
        charge_stack    = []
        
        # loop over the frames
        for i in range(len(img)):
            print('Computing frame %.0f'%(i+1)+os.sep+'%.0f'%(len(img)))
            
            
            if not (prescribed_field is None):
                input_field = prescribed_field[i]
            else:
                input_field = None
            
            # perform detection for frame i
            e_vec, err_vec, cost_vec, theta_vec, phi, defect_char = get_anisotropy(img[i], R, sigma, bin_, fov, BoxSize, order_threshold, peak_threshold, plotit=plotit, prescribed_field=input_field, stack=False)
            # add to lists of lists
            e_stack.append(e_vec)
            err_stack.append(err_vec)
            cost_stack.append(cost_vec)
            theta_stack.append(theta_vec)
            phi_stack.append(phi)
            centroids_stack.append([defect_char['y'].to_numpy(), defect_char['x'].to_numpy()])
            charge_stack.append(defect_char['charge'])
            axis_stack.append(defect_char['axis'])
            
            # append to 1-d list
            xdf = [*xdf, *defect_char['x']]
            ydf = [*ydf, *defect_char['y']]
            edf = [*edf, *e_vec]
            errdf = [*errdf, *err_vec]
            axisdf = [*axisdf, *defect_char['axis']]
            chargedf = [*chargedf, *defect_char['charge']]
            minddf = [*minddf, *defect_char['MinDist']]
            tdf = [*tdf, *([i]*len(defect_char))]
            
            centr = [defect_char['y'], defect_char['x']]
            dist_mat = cdist(centr, centr)
            
            if plotit:
                # if plotit is true, the previous call to get_anisotropy openned a figure
                plt.savefig(savedir+os.sep+'img%.0f.png'%(i), dpi=600)
                plt.close('all')
        
        # List of anisotropy only contain +1/2 info, and then are smaller in size
        # than the target table defectdf. We intercalate nan at non-+1/2 positions 
        for i in range(len(chargedf)):
            if np.abs(chargedf[i]-1/2)>0.1:
                edf = [*edf[:i], np.nan, *edf[i:]]
                errdf = [*errdf[:i], np.nan, *errdf[i:]]
        
        # fill the output table
        defectdf['x']           = xdf
        defectdf['y']           = ydf
        defectdf['frame']       = tdf
        defectdf['charge']      = chargedf
        defectdf['axis']        = axisdf
        defectdf['Anisotropy']  = edf
        defectdf['Error']       = errdf
        defectdf['MinDist']     = minddf
        
        
        ## Perform tracking (parameters are probably wrong)
        tp.quiet()       # disable tracking display 
        
        if defectdf.empty:
            defectdf['particle'] = np.ones(len(defectdf))*np.nan
        else:
            # define tracking parameters
            memory = 5 #max(round(len(np.unique(tdf))/15), 2)
            searchR = np.mean(img.shape)/10 
            print('tracking with search_range = image size/10 = %.1f px, and memory = 5 frames'%(searchR))
            try:
                defectdf = track_by_charge(defectdf, searchR, memory)
            except tp.SubnetOversizeException:
                searchR = searchR/2
                print('hehe actually printing with search_range half of that')
                defectdf = track_by_charge(defectdf, searchR, memory)
        
        return e_stack, err_stack, cost_stack, theta_stack, phi_stack, defectdf

def analyze_image(imgpath, feature_size, R, order_threshold, prescribed_field=None, plotit=False, stack=False, savedir = None, give_field=False):
    """
    Interface allows to tune 3 parameters and get_anisotropy takes 6.
    This function couples the 6 get_anisotropy parameters to the 3 inputs.
    
    See documentation for get_anisotropy
    """
    
    # Coupling
    bin_    = round(feature_size/4)
    sigma   = round(feature_size*1.5)
    
    # calls get_anisotropy
    return  get_anisotropy(imgpath, R, sigma, bin_, order_threshold=order_threshold, prescribed_field=prescribed_field, plotit=plotit, stack=stack, savedir = savedir, give_field=give_field)
    
def anisotropy_on_directory(dirname, sigma, bin_, fov, BoxSize, order_threshold, peak_threshold, R, plotf = True):
    """
    Apply the get_anisotropy function on all files in the provided folder.

    Parameters
    ----------
    dirname : string
        Path to directory on which we loop.
    sigma : float
        Size of the matrix on which we compute the structure tensor in order
        to compute the director field of the image.
    bin_ : int
        Downsampling of the director field with respect to the pixels in image.
    fov : float
        Standard deviation of the gaussian filter used to compute the average 
        order parameter on the director field. In unit of director field.
    BoxSize : int
        Distance to defect around which we take a rectangular contour, used 
        to compute the charge of the defect.
    order_threshold : float
        Threshold for order parameter. If the order is lower than this threshold
        the algorithm detects a defect.
    peak_threshold : float
        Angular threshold used to detect a jump in the angle on the contour
        determined with BoxSize. Each jump adds +1/2 or -1/2 to the charge.
    R : float
        Defined the radius around the defect used to compute the angular profile
        characteristic of a certain anisotropy.
    plotf : Bool, optional
        If True, plots representing the many computation steps of the process. 
        The default is True.

    Returns
    -------
    e_vec, costs, e_profile_av, np.mean(e_field_av)
    e_vec : list of float
        List of the anisotropy of all the +1/2 defects found.
    costs : list of float
        Lists the corresponding cost (intergated difference with theoretical curves)
        corresponding to above anisotropies.
    e_profile_av : 
        For each file, anisotropy associated with an average angular profile.
    e_mean :
        Mean of these anisotropies.

    """
    
    e_vec = []
    th_vec = []
    e_field_av = []
    
    for fname in os.listdir(dirname):
        e_, err_vec, cost_vec, theta_, phi, _ = get_anisotropy(dirname+os.sep+fname, R, sigma, bin_, fov, BoxSize, order_threshold, peak_threshold, plotit=False, stack=False)
        # e_field, err_vec, cost_vec, theta_field, phi = get_anisotropy(dirname+os.sep+fname, True, R, sigma, bin_, fov, BoxSize, order_threshold, peak_threshold, plotit=False, stack=False)
        e_vec = [*e_vec, *e_]
        th_vec = [*th_vec, *theta_]
        # e_field_av.append(e_field)
        #print(theta_)
        #for i in range(len(theta_)): print(len(theta_[i]))
        
        
    e_av = np.nanmean(e_vec)
    e_std = np.nanstd(e_vec)
    #for i in range(len(th_vec)): print(th_vec[i]);print(len(th_vec[i]))
    th_vec = np.array(th_vec)
    th_av = np.arctan2(np.nanmean(np.sin(2*th_vec), axis=0), np.nanmean(np.cos(2*th_vec), axis=0))/2
    nem_mean_y = np.nanmean(np.sin(2*th_vec), axis=0)
    nem_mean_x = np.nanmean(np.cos(2*th_vec), axis=0)
    th_av = np.arctan2(nem_mean_y, nem_mean_x)/2
    nem_std_y = np.nanstd(np.sin(2*th_vec), axis=0)
    nem_std_x = np.nanstd(np.cos(2*th_vec), axis=0)
    th_std = 1/(np.abs(nem_mean_x)*(1+(nem_mean_y/nem_mean_x)**2))*np.sqrt(nem_std_y**2 + (nem_mean_y/nem_mean_x*nem_std_x)**2) # using the formula of error propagation. I assumed cos(2x) and sin(2x) were uncorrelated though
    
    es, costs = anisotropy_comparison(phi, th_av)
    e_profile_av = es[np.argmin(costs)]
    
    if plotf:
        th_av[phi>np.pi/4] = th_av[phi>np.pi/4]%(np.pi)
        # theta_field[phi>np.pi/4] = theta_field[phi>np.pi/4]%(np.pi)

        ref_av = reference_profile(e_av)
        ref_profile = reference_profile(e_profile_av)
        ref_field = reference_profile(np.nanmean(e_field_av))
        std_ref_up = reference_profile(min(1,e_av+e_std))
        std_ref_down = reference_profile(max(-1,e_av-e_std))
        
        plt.figure()
        plt.plot(phi, ref_av, '--', label='Individual average:\n$e=%.2f\\pm%.2f$'%(e_av, e_std))
        plt.errorbar(phi, th_av, th_std, fmt = '.', label='Average profile')
        plt.plot(phi, ref_profile, '--', color=plt.gca().lines[-1].get_color(), label=r'Reference for e=%.2f'%(e_profile_av))
        # plt.plot(phi, theta_field, '.', label='Profile of average field')
        plt.plot(phi, ref_field, '--', color=plt.gca().lines[-1].get_color(), label=r'Reference for e=%.2f'%(np.mean(e_field_av)))
        
        plt.xlabel('Azimuthal Angle [rad]')
        plt.ylabel('Director angle [rad]')
        plt.legend()
        plt.tight_layout()
        
        colorm = cm.get_cmap('OrRd')
        maxdev = 3*np.std(e_vec)
        em = np.mean(e_vec)
        plt.figure()
        for i in range(len(e_vec)):
            plt.plot(phi, th_vec[i], '.', color=colorm(np.abs(e_vec[i]-em)/maxdev))
        plt.plot(phi, ref_av, 'k-', label='Mean e profile')
        plt.plot(phi, std_ref_up, 'k--', label='e$\\pm$std profiles')
        plt.plot(phi, std_ref_down, 'k--')
        plt.plot([], [], 'k.', label='Data')
        plt.xlabel('Azimuthal Angle [rad]')
        plt.ylabel('Director angle [rad]')
        plt.colorbar(cm.ScalarMappable(norm=Normalize(0, maxdev), cmap='OrRd'), label='deviation to mean anisotropy')
        plt.legend()
        plt.tight_layout()
    #print('one step')
    return e_vec, costs, e_profile_av, np.mean(e_field_av)
            
            
            

    
        