# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 18:28:44 2024

@author: Quentin

This script contains only one functions: defect_analyzer.
It's the highest function of the hierarchy: it only treats interface
"""
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf
from matplotlib import cm
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
from matplotlib.backends.backend_agg import FigureCanvasAgg
import trackpy as tp
import os

origin_file = os.path.abspath( os.path.dirname( __file__ ) )

def plot_defect_map(centroids, chargedef, axisdef, img = [], xfield= [], yfield = [], vfield = [], es = [], cimg='binary', cfield='forestgreen', e_map = 'PiYG', cmdef = 'b', cintp = 'orange', cintm = 'purple', cother = 'k'):
    """
    Plot a map with defects associated with provided informations

    Parameters
    ----------
    centroids : list of 2 1D arrays
        Coordinates of the defects to plot. In pixels.
    chargedef : list of floats
        Charge of the defects to plot.
    axisdef : list of floats
        Axis of the defects to plot.
    img : 2D array image, optional
        Image to plot on. The default is [].
    xfield : 2D array, optional
        x-cordinate of the director arrows. The default is [].
    yfield : 2D array, optional
        y-coordinates of the director arrows. The default is [].
    vfield : 2D array, optional
        Director angles. The default is [].
    es : list of float, optional
        List of the anisotropies of the defects. The default is [].
    cimg : colormap, optional
        Colormap used to plot the image. The default is 'binary' (reversed grey).
    cfield : color, optional
        Color used to plot the director field. The default is 'forestgreen'.
    e_map : colormap, optional
        Colormap used to plot the +1/2 color, as a function of their anisotropy.
        The default is 'PiYG'.
    cmdef : color, optional
        Color used to plot -1/2 defects. The default is 'b'.
    cintp : color, optional
        Color used to plot the +1 defects. The default is 'orange'.
    cintm : color, optional
        Color used to plot the -1 defects. The default is 'purple'.
    cother : color, optional
        Color used to plot all other defects, except the one with charge 0 which
        are not plotted. The default is 'k'.

    Returns
    -------
    f : Figure
        Figure oject containg all that is plotted.

    """
    
    f, ax  = plt.subplots()
    if len(img)>0:
        plt.imshow(img, cmap=cimg)
        sc = len(vfield)/len(img)
    else:
        plt.gca().invert_yaxis()
        sc = 1
        
    if len(vfield)>0:
        plt.quiver(xfield, yfield, np.cos(vfield), np.sin(vfield), angles = 'xy', scale = sc, 
                   headaxislength=0, headlength=0, pivot='mid', 
                   color=cfield, units='xy')
    
    plot_cbar = bool(len(es))
    with_anisotropy = False
    if plot_cbar:
        if len(es)==1:
            if es[0]!=np.nan:
                with_anisotropy=True
        else:
            with_anisotropy = True
        
        
    #with_anisotropy = plot_cbar
    colorm = cm.get_cmap(e_map)
    
    indent = 0
    
    lim = 0.5
    
    
    for i in range(len(chargedef)):
        if np.abs(chargedef[i]-1/2)<0.1:
            if with_anisotropy:
                c = colorm(es[i]/2/lim+0.5)
                ax.annotate('%.2f'%(es[i]), (centroids[i,1]+sc, centroids[i,0]+sc),
                         color = c, fontsize='small', path_effects=[pe.withStroke(linewidth=1, foreground="k")])
                indent += 1
            else:
                c = 'r'    
            ax.quiver(centroids[i,1], centroids[i,0], np.cos(axisdef[i]), np.sin(axisdef[i]), angles='xy', color=c, edgecolor='k', linewidth=1)
        elif np.abs(chargedef[i]+1/2)<0.1:
            ax.quiver(centroids[i,1], centroids[i,0], np.cos(axisdef[i]), np.sin(axisdef[i]), angles='xy', color='b')
            ax.quiver(centroids[i,1], centroids[i,0], np.cos(axisdef[i]+2*np.pi/3), np.sin(axisdef[i]+2*np.pi/3), angles='xy', color=cmdef)
            ax.quiver(centroids[i,1], centroids[i,0], np.cos(axisdef[i]-2*np.pi/3), np.sin(axisdef[i]-2*np.pi/3), angles='xy', color=cmdef)
        elif np.abs(chargedef[i]+1)<0.1:
            ax.plot(centroids[i,1], centroids[i,0], 'o', color = cintm)
        elif np.abs(chargedef[i]-1)<0.1:
            ax.plot(centroids[i,1], centroids[i,0], 'o', color = cintp)
        #else:
            #plt.plot(centroids[i,1], centroids[i,0], 'o', color = cother)
    
    if plot_cbar:
        plt.colorbar(cm.ScalarMappable(norm=Normalize(-lim, lim), cmap=e_map), label='Anisotropy []', ax=ax)
        
    return f

def plot_profiles(theta, e_vec, err_vec, individual = False):
    """
    Not sur I remember
    Plot the angular pofile and the associated theoretical profile.

    Parameters
    ----------
    theta : 1D array
        Director angles of the profiles, corresponding to the azimuthal angle 
        stored in ref_epsilon/orientationAzimuthal.npy
    e_vec : float
        DESCRIPTION.
    err_vec : TYPE
        DESCRIPTION.
    individual : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    phi = np.load('.'+os.sep+'ref_epsilon'+os.sep+'orientationAzimuthal.npy')
    
    if individual:
        fs = []
        for i in range(len(e_vec)):
            refth = np.load('.'+os.sep+'ref_epsilon'+os.sep+'orientationTheta_e%.2f.npy'%(e_vec[i]))
            thpstd = np.load('.'+os.sep+'ref_epsilon'+os.sep+'orientationTheta_e%.2f.npy'%(min(1,e_vec[i]+err_vec[i])))
            thmstd = np.load('.'+os.sep+'ref_epsilon'+os.sep+'orientationTheta_e%.2f.npy'%(max(-1,e_vec[i]-err_vec[i])))
            
            f = plt.figure()
            plt.plot(phi, theta[i], 'o')
            plt.plot(phi, refth, '-')
            c = plt.gca().lines[-1].get_color()
            plt.plot(phi, thmstd, '--', color = c)
            plt.plot(phi, thpstd, '--', color = c)
            plt.title(r'$e=%.2f\\pm%.2f$'%(e_vec[i], err_vec[i]))
            plt.xlabel(r'Azimuthal angle $\phi$ [rad]')
            plt.ylabel(r'Director angle $\theta$ [rad]')
            plt.title('Fit $e=%.2f\\pm%.2f'%(e_vec[i], err_vec[i]))
            plt.tight_layout()
            fs.append(f)
    
        colorm = cm.get_cmap('OrRd')
        maxdev = 3*np.std(e_vec)
        em = np.mean(e_vec)
        f2 = plt.figure()
        
        ref_av = np.load('.'+os.sep+'ref_epsilon'+os.sep+'orientationTheta_e%.2f.npy'%(em))
        std_ref_up = np.load('.'+os.sep+'ref_epsilon'+os.sep+'orientationTheta_e%.2f.npy'%(min(1,em+maxdev/3)))
        std_ref_down = np.load('.'+os.sep+'ref_epsilon'+os.sep+'orientationTheta_e%.2f.npy'%(max(-1,em-maxdev/3)))
        
        for i in range(len(e_vec)):
            plt.plot(phi, theta[i], '.', color=colorm(np.abs(e_vec[i]-em)/maxdev))
        plt.plot(phi, ref_av, 'k-', label='Mean e profile')
        plt.plot(phi, std_ref_up, 'k--', label='e$\\pm$std profiles')
        plt.plot(phi, std_ref_down, 'k--')
        plt.plot([], [], 'k.', label='Data')
        plt.xlabel('Azimuthal Angle [rad]')
        plt.ylabel('Director angle [rad]')
        plt.colorbar(cm.ScalarMappable(norm=Normalize(0, maxdev), cmap='OrRd'), label='deviation to mean anisotropy')
        plt.legend()
        plt.tight_layout()
        
        f3 = plt.figure()
        plt.hist(e_vec)
        plt.xlabel('Anisotropy []')
        plt.ylabel('Occurence')
        plt.title(r'$e=%.2f\pm%.2f'%(em, maxdev/3))
        plt.tight_layout()
        
        if individual:
            return fs, f2, f3
        else:
            return f2, f3

def trackmap(frame, traj, savedir=np.nan, filt=np.nan, yes_traj=True):
    ### Displays the frames with trajectories on it and saves them on the given folder.
    ### To do the movie go to FIJI or be smarter than men # well I know how now but I need time
    if not(np.isnan(filt)):
        traj = tp.filtering.filter_stubs(traj, filt)
    
    if len(frame)<np.max(traj['frame']):
        print('The stack and the trajectory dataframe do not match.')
    
    figlist = []
    framevec = traj['frame']
    for i in range(len(frame)):
        trajframe = traj[framevec<=i]
        currentframe = traj[framevec==i]
        if len(currentframe)>0:
            the_es = np.array(currentframe['Anisotropy'])
        else:
            the_es = [np.nan]
        f = plot_defect_map(np.array([currentframe['y'], currentframe['x']]).transpose(), np.array(currentframe['charge']), np.array(currentframe['axis']), img = frame[i], es=the_es)
        if yes_traj:
            particleframe = trajframe['particle']
            ntraj = np.unique(particleframe)
            for j in range(len(ntraj)):
                plt.plot(trajframe['x'][particleframe==ntraj[j]], trajframe['y'][particleframe==ntraj[j]])
        
        canvas = FigureCanvasAgg(f)
        canvas.draw()
        buf = canvas.buffer_rgba()
        img_array = np.asarray(buf)
        figlist.append(img_array[:, :, :3])
        plt.close()
    if isinstance(savedir, str):
        stack = np.stack(figlist)
        tf.imwrite(savedir+os.sep+'movie.tif', stack)
    
def plot_indexed_map(data, plotimg = []):
    f = plt.figure()
    plt.imshow(plotimg, cmap='gray')
    traj = data['particle']
    trajlist = np.unique(traj)
    for i in range(len(trajlist)):
        plt.plot(data['x'][traj==trajlist[i]], data['y'][traj==trajlist[i]])
        xlast = (data['x'][traj==trajlist[i]]).to_numpy()[-1]
        ylast = (data['y'][traj==trajlist[i]]).to_numpy()[-1]
        plt.annotate(str(trajlist[i]), (xlast+1, ylast+1), color=plt.gca().lines[-1].get_color())
    
    return f

