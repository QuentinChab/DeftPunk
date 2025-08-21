#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 18:38:14 2025

@author: quentin
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
from matplotlib import cm
from matplotlib.widgets import Button

def create_button(fig, ax_pos, text, func):
    button_ax = fig.add_axes(ax_pos) # create the axis where button will be displayed
    new_button = Button(button_ax, text, hovercolor='0.975') # create buttons with label `text`
    new_button.on_clicked(func) # func is called when button is pressed
    return new_button

def update_display(pos, fig, art_vec, R_vec, field, ax, R, dchar, bin_, fieldcolor='navy'):
    """
    When detection is updated in the detection interface.

    Parameters
    ----------
    pos : List of 2 arrays
        contain [x_coords, y_coords] coordinates of the director field points.
    fig : matplotlib Figure
        Figure on which the objects are drawn.
    art_vec : list of Objects
        Contains list of draw arrows and annotations.
        art_vec[0] is the director field
    R_vec : list of Objects
        Contains list of contour drawn around defect.
    field : 2D array
        Director field of the image.
    ax : matplotlib axis
        Axis where objects are drawn.
    R : float
        DESCRIPTION.
    dchar : Pandas DataFrame
        Table containing defect informations from detection.
    bin_ : int
        Pooling factor between image and director field.
    fieldcolor : STR, optional
        Color to draw director field. The default is 'navy'.

    Returns
    -------
    None.

    """
    ### update display
    R_vis = False # are contours of anisotropy computation drawn?
    vis = art_vec[0].get_visible() # is director field drawn 
    
    # Remove all previous display
    art_vec[0].remove()
    for i in range(1,len(art_vec)):
        if not (art_vec[i] is None):
            art_vec[i].remove()
        if not (R_vec[i-1] is None):
            R_vec[i-1].remove()
            R_vis = R_vec[i-1].get_visible()
    
    ### Draw
    # Draw director field
    art_vec[0] = ax.quiver(pos[0], pos[1], np.cos(field), np.sin(field), angles='xy', pivot='mid', headlength=0, headaxislength=0, scale_units='xy', scale=1/bin_ , color=fieldcolor, visible=vis)
    # draw defects
    art_vec_new,R_vec_new = draw_defects(ax, dchar, R=R, R_vis=R_vis)
    
    # Add all those objects to the lists. Lists are mutable so the value is updated outside the function.
    for i in range(1, max(len(art_vec), len(art_vec_new)+1)):
        if i>=len(art_vec):
            art_vec.append(art_vec_new[i-1])
            R_vec.append(R_vec_new[i-1])
        elif i>=len(art_vec_new)+1:
            art_vec[i] = None
            R_vec[i-1] = None
        else:
            art_vec[i] = art_vec_new[i-1]
            R_vec[i-1] = R_vec_new[i-1]
    
    fig.canvas.draw_idle()
    
def draw_defects(ax, all_data, frame=None, R=1, plot_cbar=False, R_vis=False):
    
    """
    Draw on ax the defects passed on defect_df
    
    
    Parameters
    ----------
    ax : axes
        Axis on which to draw the defects and annotations.
    all_data : DataFrame
        Contains defects information. It minimally has the columns
        'charge', 'Anisotropy', 'axis', 'x' and 'y'
    frame : int, optional
        frame number of the image we wish to annotate. Default is None (if not a stack)
    R : int, optional
        Radius of detection in pixel for anisotropy. Default is 1.
    plot_cbar : Bool, optional
        Do you plot the colorbar? The default is False.
    R_vis : boolean, optional
        Do we plot the contour for anisotropy detection?

    Returns
    -------
    artists_vec : list of Objects
        Objects newly drawn on the ax. It does not include R-contour
    R_vec : list of Objects
        List of new R-contours.

    """
    
    # if there is no defect there is no object to draw!
    if not bool(len(all_data)):
        return [], []
    
    # get xlim and ylim because changing axis will change display range
    current_xlim = ax.get_xlim()
    current_ylim = ax.get_ylim()
    
    this_phi = np.linspace(0, 2*np.pi, 30)
    
    # defect_df will contain data related to chosen frame
    # all_data from chosen frame and all previous ones
    if frame is None:
        defect_df = all_data
        Npart = 1
    else:
        defect_df = all_data[all_data['frame']==frame]
        all_data = all_data[all_data['frame']<=frame]
        Npart = len(np.unique(all_data['particle'])) # number of trajectories
    
    # extract data
    chargedef = np.array(defect_df['charge'])
    centroids = np.array([defect_df['y'], defect_df['x']]).transpose()
    es        = np.array(defect_df['Anisotropy'])
    axisdef   = np.array(defect_df['axis'])
    
    # arrows and annotations will be stored in artists_vec
    # length is Ndef + 2 artists per -1/2 defect, + 2 annotation per +1/2 + number of trajectories + colorbar
    artists_vec = [None]*(len(chargedef)+2*np.sum(np.abs(chargedef+0.5)<0.1)+2*np.sum(np.abs(chargedef-0.5)<0.1)+Npart+plot_cbar)
    R_vec = [None]*len(artists_vec)
    
    # because the number of objects in artists_def is higher than number of defects
    lim = 0.5
    e_map = 'PiYG'
    colorm = plt.get_cmap(e_map)
    incr = 0
    for i in range(len(chargedef)): # loop over defect
        if np.abs(chargedef[i]-1/2)<0.1:
            c = colorm(es[i]/2/lim+0.5)
            # write the anisotropy next to the defect
            # ax.plot(centroids[i,1], centroids[i,0], 'o', color='#D74E09')[0]
            artists_vec[incr] = ax.annotate('%.2f'%(es[i]), (centroids[i,1], centroids[i,0]),
                        color = c, fontsize='small', path_effects=[pe.withStroke(linewidth=1, foreground="k")])
            # draw the arrow to indicate +1/2 defect
            artists_vec[incr+1] = ax.quiver(centroids[i,1], centroids[i,0], np.cos(axisdef[i]), np.sin(axisdef[i]), angles='xy', color=c, edgecolor='k', linewidth=1)
            # draw the contour around the defect
            R_vec[i] = ax.plot(centroids[i,1]+R*np.cos(this_phi), centroids[i,0]+R*np.sin(this_phi), 'r', visible=R_vis)[0]
            incr += 3 #2
        elif np.abs(chargedef[i]+1/2)<0.1:
            minuscolor = 'cornflowerblue'
            # draw 3 arrows to indicate -1/2 in shape of tripod
            # ax.plot(centroids[i,1], centroids[i,0], 'o', color='#D74E09')[0]
            artists_vec[incr] = ax.quiver(centroids[i,1], centroids[i,0], np.cos(axisdef[i]), np.sin(axisdef[i]), angles='xy', color=minuscolor)
            artists_vec[incr+1] = ax.quiver(centroids[i,1], centroids[i,0], np.cos(axisdef[i]+2*np.pi/3), np.sin(axisdef[i]+2*np.pi/3), angles='xy', color=minuscolor)
            artists_vec[incr+2] = ax.quiver(centroids[i,1], centroids[i,0], np.cos(axisdef[i]-2*np.pi/3), np.sin(axisdef[i]-2*np.pi/3), angles='xy', color=minuscolor)
            incr+=3
            
        # any other defect is drawn as a point
        elif np.abs(chargedef[i]+1)<0.1:
            artists_vec[incr] = ax.plot(centroids[i,1], centroids[i,0], 'o', color = 'orange')
            incr += 1
        elif np.abs(chargedef[i]-1)<0.1:
            artists_vec[incr] = ax.plot(centroids[i,1], centroids[i,0], 'o', color = 'purple')
            incr += 1
        else:
            #plt.plot(centroids[i,1], centroids[i,0], 'o', color = cother)
            incr+=1
            
    # colorbar
    if plot_cbar:
        plt.colorbar(cm.ScalarMappable(norm=Normalize(-lim, lim), cmap=e_map), ax=ax, label='Splay-Bend Anisotropy []')
        #incr += 1
    
    # plot the trajectory from previous points
    if not (frame is None):
        trajs = np.unique(all_data['particle'])
        for i in range(len(trajs)): # loop over trajetory
            if not np.isnan(trajs[i]):
                indices = all_data['particle']==trajs[i]
                artists_vec[incr] = plt.plot(all_data['x'][indices], all_data['y'][indices], color='C%.0f'%(trajs[i]%10))
        
    # keep same diplay range as before
    new_xlim = ax.get_xlim()
    new_ylim = ax.get_ylim()
    if new_xlim[0]<current_xlim[0] or new_xlim[1]>current_xlim[1]: 
        ax.set_xlim(current_xlim)
    if new_ylim[0]>current_ylim[0] or new_ylim[1]<current_ylim[1]: 
        ax.set_ylim(current_ylim)
    
    return artists_vec, R_vec