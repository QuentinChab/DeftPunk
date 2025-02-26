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
from scipy.optimize import curve_fit
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
    phi = np.load(origin_file+os.sep+'ref_epsilon'+os.sep+'orientationAzimuthal.npy')
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

def anisotropy_comparison(phi, theta, R=np.nan, path = '.'+os.sep+'ref_epsilon_shift'+os.sep):#r'.\ref_epsilon\\'
    if np.all(np.isnan(theta)):
        return [np.nan], [np.nan]
    if np.isnan(R):
        path = origin_file+os.sep+'ref_epsilon'+os.sep
        es = np.load(path + 'e_vec.npy')
        phi_ref = np.load(path + 'orientationAzimuthal.npy')
        costs = np.ones(es.shape)
    else:
        path = origin_file+os.sep+'ref_epsilon'+os.sep
        es = np.load(path + 'e.npy')
        phi_ref = np.load(path + 'phi.npy')
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
        phiref = np.load('.'+os.sep+'ref_epsilon'+os.sep+'orientationAzimuthal.npy')
        thref = np.load('.'+os.sep+'ref_epsilon'+os.sep+'orientationTheta_e%.2f.npy'%(e))
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

def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=True):
    """
    Produce a circular histogram of angles on ax.
    
    from jwalton on stack overflow https://stackoverflow.com/questions/22562364/circular-polar-histogram-in-python [consulted 17/02/2025]
    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    x = (x+np.pi) % (2*np.pi) - np.pi
    
    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    n, bins = np.histogram(x[np.logical_not(np.isnan(x))], bins=bins)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                     edgecolor='C0', fill=False, linewidth=1)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return n, bins, patches

def motility_analysis(dataset, dt=1, unit_per_frame=1, unit_t = 'frame', unit_per_px = 1, unit_space = 'px'):
    datap = dataset[dataset['charge']==0.5]
    part_vec = datap['particle']
    part_list = np.unique(part_vec)
    dangle_list = [ [] for _ in range(len(part_list)) ]
    dangle_flat = []
    SD_flat = []
    
    #polar plot histogram and trajectory schematic
    
    f_traj = plt.figure()
    plt.quiver(-1,0, label='Head-to-tail direction')
    
    for i in range(len(part_list)):
        datapart = datap[datap['particle']==part_list[i]]
        vx = np.diff(datapart['x'], dt)*unit_per_px/unit_per_frame
        vy = np.diff(datapart['y'], dt)*unit_per_px/unit_per_frame
        axis = datapart['axis'].to_numpy()[:-dt]
        dangle = np.arccos((vx*np.cos(axis) + vy*np.sin(axis))/np.sqrt(vx**2+vy**2))
        dangle_list[i] = dangle
        dangle_flat = [*dangle_flat, *dangle]
        vamp = np.sqrt(vx**2 + vy**2)
        
        SD_flat = [*SD_flat, *vamp]
        #build traj
        xtraj = np.zeros(len(dangle)+1)
        ytraj = np.zeros(len(dangle)+1)
        for j in range(1,len(xtraj)):
            xtraj[j] = (xtraj[j-1]+vamp[j-1]*np.cos(dangle[j-1]))
            ytraj[j] = (ytraj[j-1]+vamp[j-1]*np.sin(dangle[j-1]))
        plt.plot(xtraj, ytraj)
        
    plt.xlabel('x ['+unit_space+']')
    plt.ylabel('y ['+unit_space+']')
    plt.legend()
    plt.title('Defect trajectory with respect to defect axis')
    plt.tight_layout()
    
    f_polar, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    circular_hist(ax, np.array(dangle_flat), bins=16, density=True, offset=0, gaps=True)
    plt.title('Angle between defect axis and velocity \n over %.0f points.\n Frequency proportionnal to box area.'%(dt))
    plt.tight_layout()
    
    # plot diffusion 
    
    Npart = [ np.sum(part_vec==part) for part in part_list]
    dt_list = np.arange(1, np.max(Npart))
    MSD = np.empty(len(dt_list))
    MSD_std = np.empty(len(dt_list))
    for it in range(len(dt_list)):
        SD_list = []
        for ip in range(len(part_list)):
            datapart = datap[datap['particle']==part_list[ip]]
            vx = np.diff(datapart['x'], dt_list[it])*unit_per_px/unit_per_frame
            vy = np.diff(datapart['y'], dt_list[it])*unit_per_px/unit_per_frame
            SD = np.square(vx) + np.square(vy)
            SD_list = [*SD_list, *SD]
        MSD[it] = np.nanmean(SD_list)
        MSD_std[it] = np.nanstd(SD_list)
    
    def linear_model(log_t, alpha, log_A):
        return log_A + log_t*alpha
    
    dt_list = dt_list*unit_per_frame
    err_RMSD = MSD_std/2/MSD
    err_logRMSD = err_RMSD/np.sqrt(MSD)
    weights = 1/np.square(err_logRMSD)
    params, cov = curve_fit(linear_model, np.log(dt_list), np.log(MSD)/2, bounds=(0,np.inf), maxfev=int(1e5))#, p0=(0.5, np.nanmean(np.log(MSD)/np.log(dt_list))))#, sigma=err_logRMSD, absolute_sigma=True, ))
    alpha, log_A = params
    alpha_err, log_A_err = np.sqrt(np.diag(cov))
    A = np.exp(log_A)
    A_err = A * log_A_err
    D = A**2 / 4
    D_err = 2*A*A_err/4
    
    # params, cov = curve_fit(linear_model, dt_list, np.sqrt(MSD), maxfev=int(1e4), p0=(0.5, np.nanmean(np.sqrt(MSD/dt_list))))#, sigma=err_RMSD, absolute_sigma=True)
    # alpha, A= params
    # alpha_err, A_err = np.sqrt(np.diag(cov))
    # D = A**2/4
    # D_err = 2*A*A_err/4
    
    fitted_RMSD = A * dt_list**alpha
    
    f_dif = plt.figure()
    plt.plot(dt_list, np.sqrt(MSD), '+', label='Data')
    plt.fill_between(dt_list, np.sqrt(MSD)-np.sqrt(MSD_std), np.sqrt(MSD)+np.sqrt(MSD_std), alpha=0.5, color=plt.gca().lines[-1].get_color())
    plt.plot(dt_list, fitted_RMSD, label='Fit: RMSD = %.3f$\\cdot\\tau^{%.3f}$'%(A, alpha))
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Time delay $\\tau$ ['+unit_t+']')
    plt.ylabel('RMS Displacement ['+unit_space+']')
    plt.legend()
    plt.title('The diffusion coefficient is $D=%.3f\\pm %.3f$\n The diffusion exponent is $\\alpha=%.3f\\pm %.3f$'%(D, D_err, alpha, alpha_err))
    plt.tight_layout()
    
    
    f_hist = plt.figure()
    plt.hist(SD_flat, bins=20)
    plt.xlabel('Velocity amplitude ['+unit_space+'/'+unit_t+']')
    plt.ylabel('Counts')
    plt.tight_layout()
    
    return [f_traj, f_polar, f_dif, f_hist]