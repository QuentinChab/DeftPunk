# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:51:53 2024

@author: Quentin
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
from OrientationPy import orientation_analysis
import random
from detect_defects import defect_detection
from math import ceil
from scipy.interpolate import interp1d
import os
plt.rcParams.update({'font.size': 16})

origin_file = os.path.abspath( os.path.dirname( __file__ ) )

def rotation(x,y,vx,vy,angle):
    
    xp = x*np.cos(angle)-y*np.sin(angle)
    yp = x*np.sin(angle)+y*np.cos(angle)
    
    ori = np.arctan2(vy, vx)
    ori = ori + angle
    vxp = np.cos(ori)
    vyp = np.sin(ori)
    
    #vxp = vx*np.cos(angle)-vy*np.sin(angle)
    #vyp = vx*np.sin(angle)+vy*np.cos(angle)
    
    return xp, yp, vxp, vyp

def compute_angle_diagram(orientation, R, center=None, axis=0, sym= False, plotthis = False):
    #Load the reference phi
    phi = np.load(origin_file+r'\ref_epsilon\orientationAzimuthal.npy')
    th_test = np.copy(orientation)
    #Create the x,y data if not provided
    s = orientation.shape
    #print(s)
    #if x is None:
    x = np.arange(0, s[1])
    y = np.arange(0, s[0])
    if (center is None) or (center[0] is None):
        #assume it is in the middle
        center = [(s[1])/2, (s[0])/2]
        
        

    #tensorx = np.cos(2*(orientation.swapaxes(0,1)))
    #tensory = np.sin(2*(orientation.swapaxes(0,1)))
    
    tensorx = np.cos(2*orientation)
    tensory = np.sin(2*orientation)
    angle_interpx = scipy.interpolate.RegularGridInterpolator((y,x), tensorx, bounds_error=False)
    angle_interpy = scipy.interpolate.RegularGridInterpolator((y,x), tensory, bounds_error=False)
    #tensor_unitx = angle_interpx((x[round(center[0])]+R*np.cos(phi), y[round(center[1])]+R*np.sin(phi)))
    #tensor_unity = angle_interpy((x[round(center[0])]+R*np.cos(phi), y[round(center[1])]+R*np.sin(phi)))
    
    tx = np.ones(phi.shape)*np.nan
    ty = np.ones(phi.shape)*np.nan
    tensor_repx = angle_interpx((center[1]+R*np.sin(phi), center[0]+R*np.cos(phi)))
    tensor_repy = angle_interpy((center[1]+R*np.sin(phi), center[0]+R*np.cos(phi)))
    #while  np.any(np.isnan(tx)) and R>2:
        # tensor_unitx = angle_interpx((y[round(center[1])]+R*np.sin(phi), x[round(center[0])]+R*np.cos(phi)))
        # tensor_unity = angle_interpy((y[round(center[1])]+R*np.sin(phi), x[round(center[0])]+R*np.cos(phi)))
    tensor_unitx = angle_interpx((center[1]+R*np.sin(phi+axis), center[0]+R*np.cos(phi+axis)))
    tensor_unity = angle_interpy((center[1]+R*np.sin(phi+axis), center[0]+R*np.cos(phi+axis)))
        #tx[np.isnan(tx)] = tensor_unitx[np.isnan(tx)]
        #ty[np.isnan(ty)] = tensor_unity[np.isnan(ty)]
        #R = R-1
    tx = tensor_unitx
    ty = tensor_unity
    
    theta_unit = ((np.arctan2(ty, tx)/2) - axis)%(np.pi)
    theta_unit[np.logical_and(phi>3*np.pi/2, theta_unit<np.pi/4)] = theta_unit[np.logical_and(phi>3*np.pi/2, theta_unit<np.pi/4)]+np.pi
    theta_unit[np.logical_and(phi<np.pi/2, theta_unit>3*np.pi/4)] = theta_unit[np.logical_and(phi<np.pi/2, theta_unit>3*np.pi/4)]-np.pi
    
    theta_rep = np.arctan2(tensor_repy, tensor_repx)/2
    '''
    offset = (phi-theta_unit)%(2*np.pi)
    tilt = phi[np.argmin(offset)]
    print(tilt)
    phi = phi-tilt
    theta_unit = theta_unit - tilt
    phi = phi%(2*np.pi)
    argind = np.argsort(phi)
    phi = phi[argind]
    theta_unit = theta_unit[argind]
    theta_unit[phi>np.pi/2] = theta_unit[phi>np.pi/2]%np.pi
      '''
    if sym:
         #cut the 2 branches
        th1 = theta_unit[phi<np.pi]
        th2 = theta_unit[phi>np.pi]
        phi1 = phi[phi<np.pi]
        phi2 = phi[phi>np.pi]
        #central symmetry around (pi, pi/2)
        th2 = np.pi-th2
        phi2 = 2*np.pi-phi2
        # average the 2 branches
        th1terp = scipy.interpolate.interp1d(phi2, th2, fill_value="extrapolate")
        th_sym = (th1terp(phi1) + th1)/2
        #put back in theta
        theta_unit[phi<np.pi] = th_sym
        theta_up = np.pi-th_sym
        phi_up = 2*np.pi - phi1
        thupinterp = scipy.interpolate.interp1d(phi_up, theta_up, fill_value="extrapolate")
        theta_unit[phi>np.pi] = thupinterp(phi[phi>np.pi])
        
    if plotthis:
        X, Y = np.meshgrid(x,y)#, indexing='ij')
        plt.figure()
        plt.gca().invert_yaxis()
        plt.quiver(X,Y, np.cos(th_test), np.sin(th_test), angles='xy', pivot='mid', scale=50, width=.003, headaxislength=0, headlength=0, color='k')
        plt.quiver(center[0]+R*np.cos(phi), center[1]+R*np.sin(phi), R*np.cos(theta_rep), -R*np.sin(theta_rep), pivot='mid', scale=500, width=.003, headaxislength=0, headlength=0, color='r')
        #plt.plot(center[0]+R*np.cos(phi), center[1]+R*np.sin(phi), 'ro')
        plt.plot(center[0], center[1], 'o')
        plt.axis('scaled')
    
    return phi, theta_unit

def anisotropy_comparison(phi, theta, R=np.nan, path = r'.\ref_epsilon_shift\\'):#r'.\ref_epsilon\\'
    if np.all(np.isnan(theta)):
        return [np.nan], [np.nan]
    if np.isnan(R):
        path = origin_file+r'\ref_epsilon\\'
        es = np.load(path + 'e_vec.npy')
        phi_ref = np.load(path + 'orientationAzimuthal.npy')
        costs = np.ones(es.shape)
    else:
        path = origin_file+r'\ref_epsilon_shift\\'
        es = np.load(path + 'e.npy')
        phi_ref = np.load(path + 'phi.npy')
        xshift= np.load(path + '/xshift.npy')
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
                th_ref = np.load(path+'R%.0f/Theta_e%.2f_xshift%.2f.npy'%(R, es[i], xshift[j]))
                if not same:
                    th_interp = scipy.interpolate.interp1d(phi_ref, th_ref)
                    th_ref = th_interp(phi)
                costs[i,j] = np.sqrt(np.nansum(np.square(np.arctan2(np.sin(2*(th_ref-theta)), np.cos(2*(th_ref-theta)))/2)))*2*np.pi/np.sum(np.logical_not(np.isnan(theta)))
        E, Shift = np.meshgrid(es, xshift)
        return E, Shift, costs

def orientation(phi, e, scratch=False, p=np.nan):
    phi = phi%(2*np.pi)
    
    if scratch:
        if np.isnan(p):
            def p_func(p):
                integrant = lambda x: np.sqrt((1+e*np.cos(2*x))/(1+p*p*e*np.cos(2*x)))
                return (np.pi + 0.5*p*scipy.integrate.quad(integrant, 0, np.pi)[0])**2
            popt = scipy.optimize.minimize_scalar(p_func, bounds=(-1/np.sqrt(np.abs(e)), 1/np.sqrt(np.abs(e))))
            p = popt.x
            
    
        def th_func(th):
            integrant = lambda x: np.sqrt((1+e*np.cos(2*x))/(1+p*p*e*np.cos(2*x)))
            return (phi - p*scipy.integrate.quad(integrant, 0, th-phi)[0])**2
        thopt = scipy.optimize.minimize_scalar(th_func)
        return thopt.x, p
    else:
        phiref = np.load('./ref_epsilon/orientationAzimuthal.npy')
        thref = np.load('./ref_epsilon/orientationTheta_e%.2f.npy'%(e))
        thinterp = interp1d(phiref, thref, fill_value='extrapolate')
        return thinterp(phi)

def make_vfield(e, N, angle=0): #make a defect field size (N+1)x(N+1)
    x = np.arange(-N/2, N/2+1)
    y = np.arange(-N/2, N/2+1)
    X,Y = np.meshgrid(x,y) # cartesian coordinates
    phi = np.arctan2(Y, X) # polar azimuthal coordinate
    theta = np.ones(phi.shape) # initialization
    
    p = np.nan
    for i in range(N+1):
        for j in range(N+1):
            theta[j,i], p = orientation(phi[j,i]+angle, e, p)
    theta = theta - angle
    vx = np.cos(theta)
    vy = np.sin(theta)
    
    return X,Y,vx,vy


def crop_and_rotate(orientation, xcenter, ycenter, axis, cropsize):
    # the xcenter/ycenter are the indices of the center
    xcenter = round(xcenter)
    ycenter = round(ycenter)
    
    sh = orientation.shape
    
    bigbox = cropsize
    lx1 = xcenter - max(0, xcenter-bigbox)
    lx2 = min(sh[1], xcenter+bigbox) - xcenter
    ly1 = ycenter - max(0, ycenter-bigbox)
    ly2 = min(sh[0], ycenter+bigbox) - ycenter
    x1 = xcenter - min(lx1, lx2)
    x2 = xcenter + min(lx1, lx2)
    y1 = ycenter - min(ly1, ly2)
    y2 = ycenter + min(ly1, ly2)
    
    xcrop_ = np.arange(x1-xcenter,x2-xcenter)
    ycrop_ = np.arange(y1-ycenter,y2-ycenter)
    xcrop, ycrop = np.meshgrid(xcrop_, ycrop_)
    piece_defect = orientation[y1:y2, x1:x2]
    
    
    nemx = np.cos(piece_defect)
    nemy = np.sin(piece_defect)
    
    rotx = scipy.ndimage.rotate(nemx, axis*180/np.pi, reshape=False, cval=np.nan)
    roty = scipy.ndimage.rotate(nemy, axis*180/np.pi, reshape=False, cval=np.nan)
    rot_angle = np.arctan2(roty, rotx)-axis
    
    # xp, yp, vxp, vyp = rotation(xcrop, ycrop, np.cos(piece_defect), np.sin(piece_defect), -axis)
    
    # bigbox = cropsize/2
    
    # xrot_ = np.arange(-bigbox,+bigbox)
    # yrot_ = np.arange(-bigbox,+bigbox)
    # xrot, yrot = np.meshgrid(xrot_, yrot_)
    
    # th_temp = np.arctan2(vyp,vxp)
    # vxtemp = np.cos(2*th_temp)
    # vytemp = np.sin(2*th_temp)
    # #vxrot = scipy.interpolate.griddata((xp.reshape(-1),yp.reshape(-1)), vxtemp.reshape(-1), (xrot, yrot))
    # #vyrot = scipy.interpolate.griddata((xp.reshape(-1),yp.reshape(-1)), vytemp.reshape(-1), (xrot, yrot))
    # vxrot = scipy.interpolate.griddata((xp.reshape(-1),yp.reshape(-1)), vxtemp.reshape(-1), (xrot, yrot))
    # vyrot = scipy.interpolate.griddata((xp.reshape(-1),yp.reshape(-1)), vytemp.reshape(-1), (xrot, yrot))
    
    # rot_angle = np.arctan2(vyrot, vxrot)/2
    
    return xcrop, ycrop, np.cos(rot_angle), np.sin(rot_angle)