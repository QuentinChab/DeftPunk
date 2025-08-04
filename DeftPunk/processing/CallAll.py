# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:04:20 2024

@author: Quentin

These functions process the different steps of the analysis but taking up inputs
from user or interface functions and call processing functions from compute_anisotropy

"""

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np
import os
import tifffile as tf
import pandas as pd
from scipy.spatial.distance import cdist
import trackpy as tp
from math import floor
from .OrientationPy import orientation_analysis
from .detect_defects import defect_detection
from .compute_anisotropy import compute_angle_diagram, anisotropy_comparison, track_by_charge, reference_profile

plt.rcParams.update({'font.size': 16})



def one_defect_anisotropy(field, R=np.nan, xc=None, yc=None, axis = 0, err = 0.05, plotit=False, sym=False, ML=False):
    """
    This function takes an orientation field for which we assume a +1/2 defect
    at given position, and compute its anisotropy.

    emin, err_e, costmin, th_min = one_defect_anisotropy(field, R)

    Parameters
    ----------
    field : 2D array 
        Angular field of a centered +1/2 defect.
    R : float, optional
        Radius of detection for angular profile. The default is np.nan.
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
        arbitrary number used to estimate an error. The default is 0.05.
    plotit : bool, optional
        If True, plot the field, defect, contour on which the angular profile
        is computed. Also plot the angular profile.
        Defautl is False.
    sym : bool, optional
        If True, mirror the angular profile, and avergae the profile and its
        mirror. It's an attempt to reduce noise and does not really work.
        Default is False.

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
    
    if np.isnan(R): #if R is not provided, scan on the maximum range of Rs
        sor = field.shape
        R_vec = np.arange(2, (min(sor)-1)/2-1)
    else:
        R_vec = [R]
        
    phi = np.load('DeftPunk'+os.sep+'processing'+os.sep+'ref_epsilon'+os.sep+'orientationAzimuthal.npy')
    
    emin = np.nan
    es_min = np.nan
    costmin = np.inf
    costs_min = np.nan
    th_min = np.ones(phi.shape)*np.nan
    err_e = np.nan
    for i in range(len(R_vec)):
        
        phi_cycle, th_cycle = compute_angle_diagram(field, R_vec[i], center=[xc, yc], axis=axis, plotthis=plotit, sym=sym)
        # attempt a tilt correction. Note that this is very wrong if the center is wrongly detected
        #ªth_cycle = th_cycle-th_cycle[0]
        es, costs = anisotropy_comparison(phi_cycle, th_cycle)
        imin = np.argmin(costs)
        if costs[imin] < costmin:
            costmin = costs[imin]
            emin = es[imin]
            th_min = th_cycle
            es_min = es
            costs_min = costs
            
            err_level = costs[imin]*(1+err)#◘ + err*(costs.max()-costs.min())
            if imin==0:
                ierr1 = 0
            else: 
                ierr1 = np.argmin(np.abs(costs[:imin]-err_level))
            if imin ==len(costs):
                ierr2 = imin
            else:
                ierr2 = imin + np.argmin(np.abs(costs[imin:]-err_level))
        
            err_e = (es[ierr2] - es[ierr1])/2
    
    if plotit:
        
        plt.figure()
        
        plt.subplot(1,2,1)
        plt.plot(es_min, costs_min,'+')
        yl = plt.ylim()
        plt.plot([emin-err_e, emin-err_e], yl, 'k--')
        plt.plot([emin+err_e, emin+err_e], yl, 'k--')
        plt.xlabel('Anisotropy []')
        plt.ylabel('Cost [rad]')
        plt.tight_layout()
        
        plt.subplot(1,2,2)
        plt.plot(phi_cycle, th_min, 'o', label='Measure')
        th_ref = np.load(os.path.abspath('..')+os.sep+'ref_epsilon'+os.sep+'orientationTheta_e%.2f.npy'%(emin))
        plt.plot(phi_cycle, th_ref, '--', label='Reference')
        #plt.plot(phi_cycle, phi_cycle)
        plt.xlabel('Azimuthal angle [rad]')
        plt.ylabel('Director angle [rad]')
        plt.title(r'e=%.2f'%(emin))
        plt.legend()
        plt.tight_layout()
        
    return emin, err_e, costmin, th_min
   
def get_anisotropy(imgpath, average=False, R=np.nan, sigma=25, bin_=None, fov=2, BoxSize=6, order_threshold=0.25, peak_threshold=0.75, prescribed_field=None, plotit=False, stack=False, savedir = None, give_field=False):
    """
    This function takes the path of an image, compute the orientation field, 
    finds the defect and estimates the naisotropy of the +1/2 types
    
    copy-paste:
    e_vec, err_vec, cost_vec, theta_vec, phi, defect_char = get_anisotropy(imgpath, False, R, sigma, bin_, fov, BoxSize, order_threshold, peak_threshold, plotit=False, stack=False, savedir = None)

    Parameters
    ----------
    imgpath : string
        Path to the image.
    average : boolean, optional
        Do we average all found +1/2 ? The default is False.
    R : number, optional
        Radius of detection for anisotropy computation. By default (of NaN), 
        the code scans several of them and chooses based on cost.
    sigma : int, optional
        Averaging window for field computation. The default is 25.
    bin_ : int, optional
        binning of the field wrt image size. The default is 10.
    fov : int, optional
        Averaging window (on the field) for defect detection. The default is 2.
    BoxSize : int, optional
        Box size for defect charge computation. The default is 8.
    order_threshold : float, optional
        Threshold of order parameter to locate defect. The default is 0.25.
    peak_threshold : float, optional
        Threshold on angle jump to find charge. The default is 0.85.
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
    
    if isinstance(imgpath, str):
        if imgpath[-3:]=='tif':
            img = tf.imread(imgpath)
        else:
            img = plt.imread(imgpath)
        if stack:
            if img.ndim>3:
                img = np.nanmean(img, axis=3)
            defectdf = pd.DataFrame()
            xdf = []
            ydf = []
            tdf = []
            edf = []
            errdf = []
            chargedf = []
            axisdf = []
            
            e_stack = []
            err_stack = []
            cost_stack = []
            theta_stack = []
            phi_stack = []
            centroids_stack = []
            axis_stack = []
            charge_stack = []
            frames = []
            for i in range(len(img)):
                print('Computing frame %.0f'%(i+1)+os.sep+'%.0f'%(len(img)))
                if not (prescribed_field is None):
                    input_field = prescribed_field[i]
                else:
                    input_field = None
                e_vec, err_vec, cost_vec, theta_vec, phi, defect_char = get_anisotropy(img[i], average, R, sigma, bin_, fov, BoxSize, order_threshold, peak_threshold, plotit=plotit, prescribed_field=input_field, stack=True)
                e_stack.append(e_vec)
                err_stack.append(err_vec)
                cost_stack.append(cost_vec)
                theta_stack.append(theta_vec)
                phi_stack.append(phi)
                centroids_stack.append([defect_char['y'].to_numpy(), defect_char['x'].to_numpy()])
                charge_stack.append(defect_char['charge'])
                axis_stack.append(defect_char['axis'])
                
                
                xdf = [*xdf, *defect_char['x']]
                ydf = [*ydf, *defect_char['y']]
                edf = [*edf, *e_vec]
                errdf = [*errdf, *err_vec]
                axisdf = [*axisdf, *defect_char['axis']]
                chargedf = [*chargedf, *defect_char['charge']]
                tdf = [*tdf, *([i]*len(defect_char))]
                
                centr = [defect_char['y'], defect_char['x']]
                dist_mat = cdist(centr, centr)
                
                if plotit:
                    plt.savefig(savedir+os.sep+'img%.0f.png'%(i), dpi=600)
                    
                    #ax = plt.gca()
                    #fig = plt.gcf()
                    #ax.axis('off')
                    #ax.margin(0)
                    #fig.tight_layout(pad=0)
                    #fig.canvas.draw()
                    #image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    #image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    #frames.append(image_from_plot)
                    
                    #uniqueid = str(datetime.datetime.now()).replace(' ', '')
                    #os.mkdir('images_'+uniqueid)
                    #plt.savefig('images_'+uniqueid+'/im%.0f.png'%(i))
                    plt.close('all')
            for i in range(len(chargedf)):
                if np.abs(chargedf[i]-1/2)>0.1:
                    edf = [*edf[:i], np.nan, *edf[i:]]
                    errdf = [*errdf[:i], np.nan, *errdf[i:]]
            defect_char_stack = [centroids_stack, charge_stack, axis_stack]
            defectdf['x'] = xdf
            defectdf['y'] = ydf
            defectdf['frame'] = tdf
            defectdf['charge'] = chargedf
            defectdf['axis'] = axisdf
            defectdf['Anisotropy'] = edf
            defectdf['Error'] = errdf
            
            # compute the distance to closest neighbor
            closest_neighbor = np.ones(len(tdf))*np.nan
            index = 0 # indexes the defect number
            tdf = np.array(tdf)
            
            for i in range(len(img)): # loop on frames
                ndef = np.sum(tdf==i)
                #print('frame %.0f has %.0f defects'%(i, ndef))
                if ndef>2: #if more than 2 defects on the frame
                    coord = np.array(centroids_stack[i]).T
                    dist_mat = cdist(coord, coord) #distance matrix for all defects in the frame
                    for j in range(len(dist_mat)):
                        if j==0:
                            closest_neighbor[index] = np.min(dist_mat[0, 1:])
                        elif j==len(dist_mat)-1:
                            closest_neighbor[index] = np.min(dist_mat[j, :-1])
                        else:
                            closest_neighbor[index] = min(np.min(dist_mat[j, :j]), np.min(dist_mat[j,j+1:])) 
                        index = index + 1
                elif ndef==2:
                    coord = np.array(centroids_stack[i]).T
                    dist_mat = cdist(coord, coord)
                    closest_neighbor[index] = dist_mat[0][1]
                    closest_neighbor[index+1] = dist_mat[0][1]
                    index = index + 2
                elif ndef==1:
                    closest_neighbor[index] = np.nan
                    index = index + 1
            defectdf['MinDist'] = closest_neighbor
            tp.quiet()
                        
            
            if defectdf.empty:
                defectdf['particle'] = np.ones(len(defectdf))*np.nan
            else:
                memory = max(round(len(np.unique(tdf))/15), 2)
                if np.all(np.isnan(closest_neighbor)):
                    searchR = np.mean(img.shape)/2 # in case we sometimes detect only one defect we take the distance as 20% of the image size
                    print('tracking with image size/2: %.1f'%(searchR))
                else:
                    searchR = 2*np.nanmin(closest_neighbor) # otherwise use the distance to closest neighbor as metric
                    print('tracking with closest neighbor*2: %.1f'%(searchR))
                try:
                    #defectdf = tp.link_df(defectdf, search_range=searchR, memory=memory, pos_columns=['x', 'y'])
                    defectdf = track_by_charge(defectdf, searchR, memory)
                except tp.SubnetOversizeException:
                    searchR = searchR/2
                    print('hehe actually printing with half of that')
                    #defectdf = tp.link_df(defectdf, search_range=searchR, memory=memory, pos_columns=['x', 'y'])
                    defectdf = track_by_charge(defectdf, searchR, memory)
            
            return e_stack, err_stack, cost_stack, theta_stack, phi_stack, defectdf#defect_char_stack
                
        elif img.ndim>2:
            img = np.mean(img[:,:,:2], axis=2) # if rgb image, average the 3 channels to get greyscale image
    else:
        img = imgpath
 
    #for ni 
    if prescribed_field is None:
        orientation, coherency, energy, x, y = orientation_analysis(img, sigma, bin_, plotit)        
    else:
        orientation = prescribed_field
        coherency = np.ones(prescribed_field.shape)
        sh = orientation.shape
        x_ = np.arange(sh[1])
        y_ = np.arange(sh[0])
        x, y = np.meshgrid(x_, y_)     
    Qloc, boxes, chargeb, defect_axis, centroidsN = defect_detection(orientation, coherency, fov, BoxSize, order_threshold, peak_threshold, plotall=plotit, method='weighted')
    
    # ok_dist will be 0/False if the point has any other defect at a distance of 1.5R, 1/True otherwise
    dist_mat = cdist(centroidsN, centroidsN)
    mindist = 2*R 
    ok_dist = np.empty(chargeb.shape)
    for i in range(len(ok_dist)):
        ok_dist[i] = np.all(np.delete(dist_mat[:,i],i)>mindist)
    
    if not (img is None):
        N = img.shape
        x1 = round((bin_ + N[0]-bin_*floor(N[0]/bin_))/2) # To be like OrientationJ. 
        y1 = round((bin_ + N[1]-bin_*floor(N[1]/bin_))/2)
        img_centroids = np.empty(centroidsN.shape)
        img_centroids[:,0] = centroidsN[:,0]*bin_+x1
        img_centroids[:,1] = centroidsN[:,1]*bin_+y1
    else:
        img = np.ones(prescribed_field.shape)*np.nan
    

    phi = np.load('DeftPunk'+os.sep+'processing'+os.sep+'ref_epsilon'+os.sep+'orientationAzimuthal.npy')
                
    fields = []
    e_vec = []#np.empty((np.sum(np.abs(chargeb-0.5)<0.1),1))*np.nan
    err_vec = []#np.empty(e_vec.shape)
    cost_vec = []#np.empty(e_vec.shape)
    fields = []
    theta_vec = []
    index = 0
    
    for i in range(len(chargeb)):
        if np.abs(chargeb[i]-0.5)<0.1:
            cropsize = max(30, 2*R+2)
            
            #rshift = 0
            #xr, yr, vxr, vyr = fan.crop_and_rotate(orientation, centroidsN[i,1]+rshift*np.cos(defect_axis[i][0]), centroidsN[i,0]+rshift*np.sin(defect_axis[i][0]), defect_axis[i][0], cropsize)#-0.9612995538149829
            #fields.append(np.arctan2(vyr, vxr))
            
            
            if not(average):
                #e_vec[index], err_vec[index], cost_vec[index], th = one_defect_anisotropy(fields[-1], R, plotit=plotit)
                #thisfield = fields[-1]
                #e_vec_i, err_vec_i, cost_vec_i, th = one_defect_anisotropy(thisfield, R, plotit=plotit)
                e_vec_i, err_vec_i, cost_vec_i, th = one_defect_anisotropy(orientation, R, xc=centroidsN[i,1], yc=centroidsN[i,0], axis=defect_axis[i], plotit=plotit)
                e_vec.append(e_vec_i)
                err_vec.append(err_vec_i)
                cost_vec.append(cost_vec_i)
                theta_vec.append(th)
                index+=1
    
    if average:
        fields = np.array(fields)
        field_av = np.arctan2(np.nanmean(np.sin(2*fields), axis=0), np.nanmean(np.cos(2*fields), axis=0))/2
        if len(fields)>0:
            e_vec, err_vec, cost_vec, theta_vec = one_defect_anisotropy(field_av, R, plotit=plotit)
        else:
            e_vec = np.nan
            err_vec = np.nan
            cost_vec = np.nan
    else:
        e_vec = np.array(e_vec).reshape(-1)
        err_vec = np.array(err_vec).reshape(-1)
        cost_vec = np.array(cost_vec).reshape(-1)
    if plotit:
        indent = 0
        
        frange = plt.figure()
        plt.imshow(img, cmap='gray')
        plt.quiver(x,y,np.cos(orientation),np.sin(orientation),angles='xy',pivot='mid',headaxislength=0,headlength=0,scale=50)
        
        fmap = plt.figure()
        plt.imshow(img, cmap='gray')
        #plt.quiver(x,y,np.cos(orientation), np.sin(orientation), angles='xy', headaxislength=0, headlength=0, pivot='mid', width=1, units='xy', scale=1/bin_, color='forestgreen')
        mycmap = 'PiYG'#'copper'#
        #☺colorm = cm.get_cmap('OrRd')
        colorm = cm.get_cmap(mycmap)
        em = np.nanmean(e_vec)
        #maxdev = 3*np.std(e_vec)#☻np.max(np.abs(em-e_vec))
        for i in range(len(chargeb)):
            if chargeb[i]==0.5:
                #c = colorm(np.abs(e_vec[indent]-em)/maxdev)
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
        #plt.colorbar(cm.ScalarMappable(norm=Normalize(0, maxdev), cmap='OrRd'), label='deviation to mean anisotropy')
        plt.colorbar(cm.ScalarMappable(norm=Normalize(-0.5, 0.5), cmap=mycmap), label='Anisotropy []')
    
    
    if len(fields)==0:
        if average:
            theta_vec = np.ones(phi.shape)*np.nan
        else:
            theta_vec = [np.ones(phi.shape)*np.nan]
        #print('ici')
    
    defect_char = pd.DataFrame()
    defect_char['charge'] = chargeb
    defect_char['axis'] = defect_axis
    defect_char['x'] = centroidsN[:,1]*bin_
    defect_char['y'] = centroidsN[:,0]*bin_
    defect_char['Anisotropy'] = np.nan
    defect_char['Error'] = np.nan
    incr = 0
    for di in range(len(defect_char)):
        if chargeb[di]==0.5: 
            defect_char.loc[di, 'Anisotropy'] = e_vec[incr]
            defect_char.loc[di, 'Error'] = err_vec[incr]
            incr +=1
            
    
    ####
    ndef = len(defect_char)
    closest_neighbor = np.ones(ndef)*np.nan
    centroids_stack = [defect_char['y'].to_numpy(), defect_char['x'].to_numpy()]
    
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

    defect_char['MinDist'] = closest_neighbor
    ####
    
    
    defect_char = defect_char[np.abs(defect_char['charge'])>0.2]
    defect_char = defect_char.reset_index(drop=True)
    if give_field:
        return e_vec, err_vec, cost_vec, theta_vec, phi, defect_char, orientation, [x,y]
    else:
        return e_vec, err_vec, cost_vec, theta_vec, phi, defect_char

def analyze_image(imgpath, feature_size, R, order_threshold, prescribed_field=None, plotit=False, stack=False, savedir = None, give_field=False):
    bin_    = round(feature_size/4)
    sigma   = round(feature_size*1.5)
    return  get_anisotropy(imgpath, False, R, sigma, bin_, order_threshold=order_threshold, prescribed_field=prescribed_field, plotit=plotit, stack=stack, savedir = savedir, give_field=give_field)
    
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
        e_, err_vec, cost_vec, theta_, phi = get_anisotropy(dirname+os.sep+fname, False, R, sigma, bin_, fov, BoxSize, order_threshold, peak_threshold, plotit=False, stack=False)
        e_field, err_vec, cost_vec, theta_field, phi = get_anisotropy(dirname+os.sep+fname, True, R, sigma, bin_, fov, BoxSize, order_threshold, peak_threshold, plotit=False, stack=False)
        e_vec = [*e_vec, *e_]
        th_vec = [*th_vec, *theta_]
        e_field_av.append(e_field)
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
        theta_field[phi>np.pi/4] = theta_field[phi>np.pi/4]%(np.pi)

        ref_av = reference_profile(e_av)
        ref_profile = reference_profile(e_profile_av)
        ref_field = reference_profile(np.nanmean(e_field_av))
        std_ref_up = reference_profile(min(1,e_av+e_std))
        std_ref_down = reference_profile(max(-1,e_av-e_std))
        
        plt.figure()
        plt.plot(phi, ref_av, '--', label='Individual average:\n$e=%.2f\\pm%.2f$'%(e_av, e_std))
        plt.errorbar(phi, th_av, th_std, fmt = '.', label='Average profile')
        plt.plot(phi, ref_profile, '--', color=plt.gca().lines[-1].get_color(), label=r'Reference for e=%.2f'%(e_profile_av))
        plt.plot(phi, theta_field, '.', label='Profile of average field')
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
            
            
            

    
        