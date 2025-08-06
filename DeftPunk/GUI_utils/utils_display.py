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
    button_ax = fig.add_axes(ax_pos)
    new_button = Button(button_ax, text, hovercolor='0.975')
    new_button.on_clicked(func)
    return new_button

def update_display(pos, fig, art_vec, R_vec, field, ax, R, dchar, bin_, fieldcolor='navy'):
    ### update display
    R_vis = False # are contours of anisotropy computation drawn?
    vis = art_vec[0].get_visible() # 
    
    # Remove all previous display
    art_vec[0].remove()
    for i in range(1,len(art_vec)):
        if not (art_vec[i] is None):
            art_vec[i].remove()
        if not (R_vec[i-1] is None):
            R_vec[i-1].remove()
            R_vis = R_vec[i-1].get_visible()
    
    # Draw director field
    art_vec[0] = ax.quiver(pos[0], pos[1], np.cos(field), np.sin(field), angles='xy', pivot='mid', headlength=0, headaxislength=0, scale_units='xy', scale=1/bin_ , color=fieldcolor, visible=vis)
    # draw defects
    art_vec_new,R_vec_new = draw_defects(ax, dchar, R=R, R_vis=R_vis)
    # Add all those objects to a list that is kept
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
    
def draw_defects(ax, all_data, frame=None, R=1, plot_cbar=False, animated=False, R_vis=False):
    
    """
    Draw on ax the defects passed on defect_df

    Parameters
    ----------
    ax : axes
        Axis on which to draw the defects and annotations.
    defect_df : DataFrame
        Contains defects information. It minimally has the columns
        'charge', 'Anisotropy', 'axis', 'x' and 'y'
    plot_cbar : Bool, optional
        Do you plot the colorbar? The default is False.

    Returns
    -------
    artists_vec : list of Objects
        Objects newly drawn on the ax. It does not include R-contour
    R_vec : list of Objects
        List of new R-contours.

    """
    # get xlim and ylim because changing axis will change display range
    current_xlim = ax.get_xlim()
    current_ylim = ax.get_ylim()
    
    this_phi = np.linspace(0, 2*np.pi, 30)
    
    if frame is None:
        defect_df = all_data
        Npart = 1
    else:
        defect_df = all_data[all_data['frame']==frame]
        all_data = all_data[all_data['frame']<=frame]
        Npart = len(np.unique(all_data['particle']))
        
    chargedef = np.array(defect_df['charge'])
    centroids = np.array([defect_df['y'], defect_df['x']]).transpose()
    es = np.array(defect_df['Anisotropy'])
    axisdef = np.array(defect_df['axis'])
    
    # arrows and annotations will be stored in artists_vec
    # length is Ndef + 2 artists per -1/2 defect, + 2 annotation per +1/2 + number of trajectories + colorbar
    artists_vec = [None]*(len(chargedef)+2*np.sum(np.abs(chargedef+0.5)<0.1)+2*np.sum(np.abs(chargedef-0.5)<0.1)+Npart+plot_cbar)
    R_vec = [None]*len(artists_vec)
    
    # because the number of objects in artists_def is higher than number of defects
    lim = 0.5
    e_map = 'PiYG'
    colorm = plt.get_cmap(e_map)
    incr = 0
    for i in range(len(chargedef)):
        if np.abs(chargedef[i]-1/2)<0.1:
            c = colorm(es[i]/2/lim+0.5)
            artists_vec[incr] = ax.annotate('%.2f'%(es[i]), (centroids[i,1], centroids[i,0]),
                        color = c, fontsize='small', path_effects=[pe.withStroke(linewidth=1, foreground="k")])
  
            artists_vec[incr+1] = ax.quiver(centroids[i,1], centroids[i,0], np.cos(axisdef[i]), np.sin(axisdef[i]), angles='xy', color=c, edgecolor='k', linewidth=1)
            R_vec[i] = ax.plot(centroids[i,1]+R*np.cos(this_phi), centroids[i,0]+R*np.sin(this_phi), 'r', visible=R_vis)[0]
            incr += 3
        elif np.abs(chargedef[i]+1/2)<0.1:
            minuscolor = 'cornflowerblue'
            artists_vec[incr] = ax.quiver(centroids[i,1], centroids[i,0], np.cos(axisdef[i]), np.sin(axisdef[i]), angles='xy', color=minuscolor)
            artists_vec[incr+1] = ax.quiver(centroids[i,1], centroids[i,0], np.cos(axisdef[i]+2*np.pi/3), np.sin(axisdef[i]+2*np.pi/3), angles='xy', color=minuscolor)
            artists_vec[incr+2] = ax.quiver(centroids[i,1], centroids[i,0], np.cos(axisdef[i]-2*np.pi/3), np.sin(axisdef[i]-2*np.pi/3), angles='xy', color=minuscolor)
            incr+=3
        elif np.abs(chargedef[i]+1)<0.1:
            artists_vec[incr] = ax.plot(centroids[i,1], centroids[i,0], 'o', color = 'orange')
            incr += 1
        elif np.abs(chargedef[i]-1)<0.1:
            artists_vec[incr] = ax.plot(centroids[i,1], centroids[i,0], 'o', color = 'purple')
            incr += 1
        else:
            #plt.plot(centroids[i,1], centroids[i,0], 'o', color = cother)
            incr+=1

    if plot_cbar:
        plt.colorbar(cm.ScalarMappable(norm=Normalize(-lim, lim), cmap=e_map), ax=ax, label='Splay-Bend Anisotropy []')
        #incr += 1
    
    if not (frame is None):
        trajs = np.unique(all_data['particle'])
        for i in range(len(trajs)):
            if not np.isnan(trajs[i]):
                indices = all_data['particle']==trajs[i]
                artists_vec[incr] = plt.plot(all_data['x'][indices], all_data['y'][indices], color='C%.0f'%(trajs[i]%10))
        
    # set back to old display range
    new_xlim = ax.get_xlim()
    new_ylim = ax.get_ylim()
    if new_xlim[0]<current_xlim[0] or new_xlim[1]>current_xlim[1]: 
        ax.set_xlim(current_xlim)
    if new_ylim[0]>current_ylim[0] or new_ylim[1]<current_ylim[1]: 
        ax.set_ylim(current_ylim)
    
    return artists_vec, R_vec