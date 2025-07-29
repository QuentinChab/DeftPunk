# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 18:28:44 2024

@author: Quentin

This script contains only one functions: defect_analyzer.
It's the highest function of the hierarchy: it only treats interface
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider, CheckButtons, TextBox
import pandas as pd
import compute_anisotropy as can
import anisotropy_functions as fan
import OrientationPy as OPy
import tifffile as tf
from matplotlib import cm
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
import tkinter
from tkinter import filedialog
import trackpy as tp
from matplotlib.animation import ArtistAnimation, FuncAnimation, FFMpegWriter
from scipy import stats
import scipy.io
import os

origin_file = os.path.abspath( os.path.dirname( __file__ ) )


bin_factor = 4

def defect_analyzer(imgpath, det_param, stack=True, frame=0, um_per_px=1, unit='px', vfield=None, endsave=True, savedir='Select'):
    """Calls the interface to analyze defect and their anisotropy on an image
       
    The exact choice of detection parameter is described at the end.
    
    Parameters
       ----------
       imgpath : str
           path to the image to analyze.
       det_param : array size 3
           with in order
           - w : int or float
               feature size in pixel on which most of detection parameters are chained to.
               As an example, the window size for vector field computation is sigma=1.5*w
               The other relations are described at the end.
           - R : int or float
               Radius of detection. Around a defect a contour is taken at distance R
               and the director field is taken on this contour. This is used to 
               compute defect anisotropy.
           - order_threshold : float between 0 and 1
               If there is a region with nematic order parameter inferior to 
               order_threshold, we consider that a defect is present
       stack : bool, optional
           If the image is a stack or not.
           Default is True
       frame : int, optional
           If the image is a stack, index of the frame displayed at the interface
           Default is 0
       um_per_px : float, optionnal
           Conversion between between pixel and the unit provided in unit parameter.
           Default is 1.
       unit : str, optionnal
           Space unit of the image.
           Default is px
       vfield : numpy array, optionnal
           User-provided director field, to skip this computation.
           If None (default) the director is computed.
       endsave : boolean, optionnal
           Do we save the data (defect location, charge, anisotropy,...)
           Default is True
       savedir : string, optionnal
           If 'Select' (default), we ask the user to browse where to save the data.
           Otherwise it is the path where data will be saved.
           
    Returns
       ----------
       defect_char : Pandas DataFrame
           Contains all the informations of detected defects. The fields are
           'charge', 'axis', 'x', 'y', 'Anisotropy' and 'Error'.
           If the image is a stack we also have the fields
           'frame', 'MinDist', 'particle'
           MinDist is the distance to the closest other defect from the same frame
           Particle is an identifier for defect tracking. 
       
       det_param : size-3 array of floats
           selected detection parameters with
           - w_out : float
               Value for feature size w chosen with the slider.
           - R_out : float
               Value for detection radius R chosen with the slider.
           - order_out : float
               Value for order_threshold parameter chosen with the slider.
       vfield : NxM numpy array
           Computed director field (binned) the size is L/int(w/4) x H/int(w/4)
           with LxH the shape of the image
       handles : list of handles
           list containing the references of the sliders and buttons, so they stay active
           
    Interface
       ----------
       Sliders
       R-slider : Changes the radius of detection R
           Between 2 and len(img)/4
       order_threshold-slider : Changes the order_threshold for defect detection.
           Between 0 and 1
           Initial value is 0.8
       w-slider : Changes the w parameter.
           Between 4 and 80 px.
           
       Buttons
       Director : Show/Hide director field
       R - Detection : Show/Hide the contour on which the anisotropy is computed
       Reset : Reset the sliders value
       Save plot : Save the image of the plot in the chosen place. It opens a file explorer.
       OK : Validate the chosen values, computes the stack with them and
            open a file explorer to save dataframe. If you do not want, close it.
            
    Detection parameters
       ----------
       sigma : 1.5*w
           Size of the filter on which the vector field is computed
       bin_ : w/4
           Downsampling factor between the image (pixels) and the vector field
       fsig : 2
           Size of the filter on which is computed the order parameter (in number of director arrows)
       order_threshold : 0.4*fsig=0.8
           Threshold for order parameter under which we consider that we locate a defect
       BoxSize : 6
           Size of the contour for charge detection (in number of director arrows)
       peak_threshold : 0.75
           Size of angular jumps in angular profile taken as revealing a charge in defect (in pi)
       R : see parameter description
    """
       
    #initial values
    global defect_char
    global over
    
    # # tkinter needs a root window, so we open it and hide it for good
    # root = tkinter.Tk()
    # root.withdraw()
    
    w = det_param[0]
    R = det_param[1]
    
    over = False
    ### Where we define the relation between the parameters ####
    sigma = round(1.5*w) #integration size for orientation field
    bin_ = round(w/bin_factor) # Sampling size for orientation field
    fsig = 2 # in units of bin. Size of filter for order parameter
    order_threshold = det_param[2]
    BoxSize = 6
    peak_threshold = 0.75
    
    if not (vfield is None):
        lock_field = True
    else:
        lock_field = False
    
    ### Where we load the image and select the displayed frame 'frame'
    #use the right unpacking package
    field_visible = False
    if imgpath is None:
        field_visible = True
        img = None
    else:
            
        if imgpath[-3:]=='tif':
            img_st = tf.imread(imgpath)
        else:
            img_st = plt.imread(imgpath)
        
        # if it is a multichannel image (color), average the channels 
        if stack:
            if img_st.ndim>3:
                img_st = np.nanmean(img_st, axis=3) #if we have several intensity channels average them
            img = img_st[frame]
        else:
            if img_st.ndim>2:
                img_st = np.nanmean(img_st, axis=2)
            img = img_st
    
    
    
    ### Where we generate the initial display ###
    e_vec, err_vec, cost_vec, theta_vec, phi, defect_char, vfield, pos = can.get_anisotropy(img, False, R/bin_, sigma, bin_, fsig, BoxSize, order_threshold, peak_threshold, prescribed_field=vfield, plotit=False, stack=stack, savedir = None, give_field = True)
    fieldcolor = 'navy'
    my_field = [vfield, pos]
    
    fig, ax = plt.subplots()
    #image 
    if not (img is None):
        back_img = plt.imshow(img, cmap='binary')
    # vector field
    qline = ax.quiver(pos[0], pos[1], np.cos(vfield), np.sin(vfield), angles='xy', pivot='mid', headlength=0, headaxislength=0, scale_units='xy', scale=1/bin_ , color=fieldcolor)
    qline.set_visible(field_visible)
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    e_map = 'PiYG'
    colorm = plt.get_cmap(e_map)
    
    lim = 0.5 # limits of anisotropy colorbar
    
    ##### Sliders ############
    if unit=='px':
        labw = 'Feature size [px]'
        labR = "Detection\n radius [px]"
    else:
        labw = 'Feature size [px] (%.2f '%(um_per_px*w)+unit+')'
        labR = "Detection radius [px]\n(%.2f "%(um_per_px*R)+unit+")"
    # Make a horizontal slider to control the feature size w.
    axw = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    w_slider = Slider(
        ax=axw,
        label= labw,
        valmin=4,
        valmax=80,
        valinit=w,
        valstep=1
    )
    
    # Make a vertically oriented slider to control the radius o fdetection R
    axR = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
    if img is None:
        Rmax = round(len(vfield)/4)
    else:
        Rmax = round(len(img)/4)
    R_slider = Slider(
        ax=axR,
        label=labR,
        valmin=2,
        valmax=Rmax,
        valinit=R,
        valstep=1,
        orientation="vertical"
    )    
    
    # Make a vertically oriented slider to control the radius o fdetection R
    axThresh = fig.add_axes([0.23, 0.25, 0.0225, 0.63])
    Thresh_slider = Slider(
        ax=axThresh,
        label="Order parameter\n detection threshold",
        valmin=0,
        valmax=1,
        valinit=order_threshold,
        valstep=0.05,
        orientation="vertical"
        )
    
    # function that plots points
    def add_points(ax, defect_df, plot_cbar=False, R_vis=False):
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
        R_vis : Bool, optional
            Is the R-contour displayed? The default is False.

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
        
        this_phi = np.linspace(0, 2*np.pi, 30) # for R-det display
        chargedef = defect_df['charge']
        centroids = np.array([defect_df['y'], defect_df['x']]).transpose()
        es = defect_df['Anisotropy']
        axisdef = defect_df['axis']
        
        # arrows and annotations will be stored in artists_vec and range diplay in R_vec
        # it will be used to change visibility and possibly remove them
        artists_vec = [None]*(2*len(chargedef)+2*np.sum(np.abs(chargedef+0.5)<0.1))
        R_vec = [None]*(2*len(chargedef)+2*np.sum(np.abs(chargedef+0.5)<0.1))
        
        # because the number of objects in artists_def is higher than number of defects
        incr = 0
        for i in range(len(chargedef)):
            if np.abs(chargedef[i]-1/2)<0.1:
                c = colorm(es[i]/2/lim+0.5)
                artists_vec[incr] = ax.annotate('%.2f'%(es[i]), (centroids[i,1], centroids[i,0]),
                            color = c, fontsize='small', path_effects=[pe.withStroke(linewidth=1, foreground="k")])
      
                artists_vec[incr+1] = ax.quiver(centroids[i,1], centroids[i,0], np.cos(axisdef[i]), np.sin(axisdef[i]), angles='xy', color=c, edgecolor='k', linewidth=1)
                R_vec[i] = ax.plot(centroids[i,1]+R_slider.val*np.cos(this_phi), centroids[i,0]+R_slider.val*np.sin(this_phi), 'r', visible=R_vis)[0]
                incr += 2
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
                incr += 1
                artists_vec[incr] = ax.plot(centroids[i,1], centroids[i,0], 'o', color = 'purple')
            #else:
                #plt.plot(centroids[i,1], centroids[i,0], 'o', color = cother)
    
        if plot_cbar:
            plt.colorbar(cm.ScalarMappable(norm=Normalize(-lim, lim), cmap=e_map), ax=ax, label='Splay-Bend Anisotropy []')
        
        # set back to old display range
        new_xlim = ax.get_xlim()
        new_ylim = ax.get_ylim()
        if new_xlim[0]<current_xlim[0] or new_xlim[1]>current_xlim[1]: 
            ax.set_xlim(current_xlim)
        if new_ylim[0]>current_ylim[0] or new_ylim[1]<current_ylim[1]: 
            ax.set_ylim(current_ylim)
        
        return artists_vec, R_vec
    
    # Lists of objects. It will be use to change their visibility and remove them
    art, R_vec = add_points(ax, defect_char, plot_cbar=True)
    # art_vec has vector field in index 0, then arrows and annotations
    art_vec = [qline, *art]
    
    ### Where we create the buttons and interface ###
    # adjust the main plot to make room for the sliders
    fig.subplots_adjust(left=0.22, bottom=0.20)

    # draw the legend
    axlegend= fig.add_axes([0.32, 0.87, 0.5, 0.13])
    imlegend = plt.imread(os.path.split(origin_file)[0]+os.sep+'GUI_images'+os.sep+'defect_type.png')
    axlegend.imshow(imlegend)
    axlegend.axis('off')
    
    # draw the schematics of the defects with different anisotropy
    axschem= fig.add_axes([0.9, 0.19, 0.1, 0.7])
    imschem = plt.imread(os.path.split(origin_file)[0]+os.sep+'GUI_images'+os.sep+'defect_style.png')
    axschem.imshow(imschem)
    axschem.axis('off')
    
    ##### Update functions for sliders
    # The function to be called anytime a slider's value changes
    def update_w(val):
        global defect_char
        global bin_
        nonlocal vfield
        
        if not(unit=='px'):
            w_slider.label.set_text('Feature size [px] (%.2f '%(um_per_px*w_slider.val)+unit+')')
        
        # get new defects and anisotropies
        sigma = round(1.5*w_slider.val) #integration size for orientation field
        bin_ = round(w_slider.val/bin_factor) # Sampling size for orientation field
        fsig = 2 # in units of bin. Size of filter for order parameter
        order_threshold = Thresh_slider.val
        BoxSize = 6
        peak_threshold = 0.75
        
        if lock_field:
            input_field = vfield
        else:
            input_field=None
        e_vec, err_vec, cost_vec, theta_vec, phi, dchar, field, pos = can.get_anisotropy(img, False, R_slider.val/bin_, sigma, bin_, fsig, BoxSize, order_threshold, peak_threshold, prescribed_field=input_field, plotit=False, stack=False, savedir = None, give_field=True)
        vfield = field
        defect_char = dchar
        my_field[0] = field
        my_field[1] = pos
        
        
        R_vis = False
        vis = art_vec[0].get_visible()
        art_vec[0].remove()
        for i in range(1,len(art_vec)):
            if not (art_vec[i] is None):
                art_vec[i].remove()
            if not (R_vec[i-1] is None):
                R_vec[i-1].remove()
                R_vis = R_vec[i-1].get_visible()
        #ax.imshow(img, cmap='binary')
        art_vec[0] = ax.quiver(pos[0], pos[1], np.cos(field), np.sin(field), angles='xy', pivot='mid', headlength=0, headaxislength=0, scale_units='xy', scale=1/bin_ , color=fieldcolor, visible=vis)
        art_vec_new,R_vec_new = add_points(ax, defect_char, R_vis=R_vis)
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
    
    
    def update_R(val):
        field = my_field[0]
        bin_ = round(w_slider.val/bin_factor) # Sampling size for orientation field
        new_anisotropy = np.empty(len(defect_char))*np.nan
        
        if not(unit=='px'):
            R_slider.label.set_text("Detection radius [px]\n(%.2f "%(um_per_px*R_slider.val)+unit+")")
        
        for i in range(len(defect_char)):
            e_vec_i, err_vec_i, cost_vec_i, th = can.one_defect_anisotropy(field, R_slider.val/bin_, xc=defect_char['x'][i]/bin_, yc=defect_char['y'][i]/bin_, axis=defect_char['axis'][i])
            new_anisotropy[i] = e_vec_i
        defect_char['Anisotropy'] = new_anisotropy
        R_vis = False
        for i in range(1,len(art_vec)):
            if not (art_vec[i] is None):
                art_vec[i].remove()
            if not (R_vec[i-1] is None):
                R_vec[i-1].remove()
                R_vis = R_vec[i-1].get_visible()
        art_vec_new,R_vec_new = add_points(ax, defect_char, R_vis=R_vis)
        
        # Update the lists of drawn objects
        for i in range(1, max(len(art_vec), len(art_vec_new)+1)):
            if i>=len(art_vec): # If we reach the list max size, add the object
                art_vec.append(art_vec_new[i-1])
                R_vec.append(R_vec_new[i-1])
            elif i>=len(art_vec_new)+1: # If there is no more new ojects, fill with None
                art_vec[i] = None
                R_vec[i-1] = None
            else: # replace old objects with new objects
                art_vec[i] = art_vec_new[i-1]
                R_vec[i-1] = R_vec_new[i-1]
        fig.canvas.draw_idle()

    def update_order(val):
        # re-compute everything
        global defect_char
        nonlocal vfield
        sigma = round(1.5*w_slider.val) #integration size for orientation field
        bin_ = round(w_slider.val/bin_factor) # Sampling size for orientation field
        fsig = 2 # in units of bin. Size of filter for order parameter
        order_threshold = Thresh_slider.val
        BoxSize = 6
        peak_threshold = 0.75
        if lock_field:
            input_field = vfield
        else:
            input_field=None
        e_vec, err_vec, cost_vec, theta_vec, phi, dchar, vfield, pos = can.get_anisotropy(img, False, R_slider.val/bin_, sigma, bin_, fsig, BoxSize, order_threshold, peak_threshold, prescribed_field=input_field, plotit=False, stack=stack, savedir = None, give_field=True)
        defect_char = dchar
        my_field[0] = vfield
        my_field[1] = pos
        
        # get previous visibility info
        R_vis = False
        vis = art_vec[0].get_visible()
        # remove previously drawn objects
        art_vec[0].remove()
        for i in range(1,len(art_vec)):
            if not (art_vec[i] is None): # the way art_vec is coded supposedly has many None at the end
                art_vec[i].remove()
            if not (R_vec[i-1] is None): # same remark for R_vec
                R_vec[i-1].remove()
                R_vis = R_vec[i-1].get_visible()
        
        # plot new objects
        art_vec[0] = ax.quiver(pos[0], pos[1], np.cos(vfield), np.sin(vfield), angles='xy', pivot='mid', headlength=0, headaxislength=0, scale_units='xy', scale=1/bin_ , color=fieldcolor, visible=vis)
        art_vec_new,R_vec_new = add_points(ax, defect_char, R_vis=R_vis)
        # add all objects 
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
    
    
    
    # register the update function with each slider
    R_slider.on_changed(update_R)
    w_slider.on_changed(update_w)
    Thresh_slider.on_changed(update_order)

    # Create 6 `matplotlib.widgets.Button`
    reverseax = fig.add_axes([0.05, 0.025, 0.1, 0.04])
    reversebutton = Button(reverseax, 'Invert Color', hovercolor='0.975')
    resetax = fig.add_axes([0.5, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')
    OKax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    OKbutton = Button(OKax, 'OK', hovercolor='0.975')
    Fieldax = fig.add_axes([0.2, 0.025, 0.1, 0.04])
    Fieldbutton = Button(Fieldax, 'Director', hovercolor='0.975')
    Circleax = fig.add_axes([0.35, 0.025, 0.1, 0.04])
    Circlebutton = Button(Circleax, 'R - Detection', hovercolor='0.975')
    Saveax = fig.add_axes([0.65, 0.025, 0.1, 0.04])
    Savebutton = Button(Saveax, 'Save Image', hovercolor='0.975')
    
    # upadate functions for buttons
    def reset(event):
        w_slider.reset()
        R_slider.reset()
        Thresh_slider.reset()
    button.on_clicked(reset)
    
    def invert_color(event):
        if back_img.cmap(360)[0]: #if it's gray then the last color, n°360, is (1,1,1)
            back_img.set_cmap('binary')
        else:
            back_img.set_cmap('gray')
        fig.canvas.draw_idle()
    reversebutton.on_clicked(invert_color)

    def finish(event):
        global defect_char
        global over
        
        det_param[0] = w_slider.val
        det_param[1] = R_slider.val
        det_param[2] = Thresh_slider.val
        
        class Placeholder: # just to be able to use fold.name line in every case
            def __init__(self, n):
                self.name = n
        
        if endsave:
            if savedir=='Select':
                print('Where to save the data?')
                fold = filedialog.asksaveasfile(defaultextension='.csv') # the user choses a place in file explorer
            else:
                fold = Placeholder(savedir+os.sep+'data.csv')
        
        sigma = round(1.5*w_slider.val) #integration size for orientation field
        bin_ = round(w_slider.val/bin_factor) # Sampling size for orientation field
        
        over = True
        # The displayed image is just one frame, now the whole stack is computed.
        if stack:
            print('Computing the whole stack...')
            e_vec, err_vec, cost_vec, theta_vec, phi, defect_char = can.get_anisotropy(imgpath, False, R_slider.val/bin_, round(1.5*w_slider.val), round(w_slider.val/bin_factor), fsig, BoxSize, Thresh_slider.val, peak_threshold, plotit=False, stack=stack, savedir = None)
        plt.close(fig)
        #print(defect_char['x'])
        
        if endsave:
            if not fold is None:
                defect_char.to_csv(fold.name) # the DataFrame is saved as csv
                print('Saved')
            else:
                print('Done')

            
    OKbutton.on_clicked(finish)
    
    def plotField(event):
        if art_vec[0].get_visible():
            art_vec[0].set_visible(False)
        else:
            art_vec[0].set_visible(True)
       #     qline = ax.quiver(pos[0], pos[1], np.cos(vfield), np.sin(vfield), angles='xy', pivot='mid', headlength=0, headaxislength=0, scale= bin_/len(img), color='forestgreen')
        fig.canvas.draw_idle()

    Fieldbutton.on_clicked(plotField)
    
    def plotR(event):
        for i in range(len(R_vec)):
            if not (R_vec[i] is None):
                if R_vec[i].get_visible():
                    R_vec[i].set_visible(False)
                else:
                    R_vec[i].set_visible(True)
        fig.canvas.draw_idle()
        new_xlim = ax.get_xlim()
        new_ylim = ax.get_ylim()
        if new_xlim[0]<x_lim[0] or new_xlim[1]>x_lim[1]: 
            ax.set_xlim(x_lim)
        if new_ylim[0]>y_lim[0] or new_ylim[1]<y_lim[1]: 
            ax.set_ylim(y_lim)

    Circlebutton.on_clicked(plotR)
    
    def ClickSave(event):
        # create another figure
        #print(defect_char['x'])
        figsave, axsave = plt.subplots()
        plt.imshow(img, cmap='binary')
        # plot all visible features 
        if art_vec[0].get_visible():
            axsave.quiver(pos[0], pos[1], np.cos(vfield), np.sin(vfield), angles='xy', pivot='mid', headlength=0, headaxislength=0, scale_units='xy', scale=1/bin_ , color=fieldcolor)
        R_vis=False
        for i in range(len(R_vec)):
            if not R_vec[i] is None:
                R_vis = R_vec[i].get_visible()
                break
        add_points(axsave, defect_char, plot_cbar=True, R_vis=R_vis)
        
        # set the main figure display range
        axsave.set_xlim(ax.get_xlim())
        axsave.set_ylim(ax.get_ylim())
        # Write parameters as title
        axsave.set_title('feature size = %.0f px, R = %.0f px\norder threshold = %.2f'%(w_slider.val, R_slider.val, Thresh_slider.val))
        
        fold = filedialog.asksaveasfile(defaultextension='.png') # make the user choose a file location
        if fold:
            figsave.savefig(fold.name) # save figure at this location
            print('Saving calcelled')
        else:
            print('Saved')
        plt.close(figsave)
        
    Savebutton.on_clicked(ClickSave)
    
    # Throw exception if the figure is closed 
    def on_close(event):
        if not over:
            raise Exception("Program interrupted by the user (closed the figure).") 
    fig.canvas.mpl_connect('close_event', on_close)
    
    ### Where the infinite loop allow us to not be garbage collected (if I understand well) ###
    # it stops when the display window is closed: either by user or when OK is clicked on
    plt.show()
    # while plt.fignum_exists(fig.number):
    #     plt.pause(0.1)
        
    return defect_char, det_param, vfield, [OKbutton, Savebutton, Circlebutton, Fieldbutton, button, reversebutton]



def defect_statistics(df, minframe=0, maxframe=np.inf, filt=0, minspace=0):
    # This represents some usual analysis of the data
    
    
    # apply the requested filters on the dataframe
    if 'frame' in df.columns:
        df = df[np.logical_and(df['frame']>=minframe, df['frame']<=maxframe)] #filter at reduced frame range
    df = tp.filter_stubs(df, filt) #only trajectories longer than filt
    
    flist = []
    
    f1 = plt.figure()
    plt.plot(df['MinDist'], df['Anisotropy'], '.', label='Data')
    plt.plot([minspace, minspace], [-1,1], 'k--', label='Filtered defects')
    plt.xlabel('Distance to nearest neighboring defect')
    plt.ylabel('Anisotropy')
    plt.ylim([-1,1])
    plt.title('Before filtering non-isolated defects')
    plt.legend()
    plt.tight_layout()
    flist.append(f1)
    
    df = df[df['MinDist']>=minspace] #only if the closest neighbor is further away than minspace
    
    # Make the stat
    emean = np.nanmean(df['Anisotropy'])
    estd  = np.nanstd(df['Anisotropy'])
    f2 = plt.figure()
    plt.hist(df['Anisotropy'], bins=20)
    plt.xlabel('Anisotropy')
    plt.ylabel('Occurence')
    plt.xlim([-1,1])
    plt.title('Average %.0f defects: $<e>=%.2f\\pm %.2f$'%(len(df), emean, estd))
    plt.tight_layout()
    flist.append(f2)
    
    if 'frame' in df.columns:
        trajs = np.unique(df['particle'])
        etraj = np.empty(len(trajs))*np.nan
        Ltraj = np.empty(len(trajs))*np.nan
        for i in range(len(trajs)):
            etraj[i] = np.mean(df['Anisotropy'][df['particle']==trajs[i]])
            Ltraj[i] = np.sum(df['particle']==trajs[i])
        f3 = plt.figure()
        plt.hist(etraj, bins=20)
        plt.xlabel('Avergae anisotropy on a trajectory')
        plt.ylabel('Occurence')
        plt.xlim([-1,1])
        plt.title('Average over %.0f traj: $<e>=%.2f\\pm %.2f$'%(len(trajs), np.nanmean(etraj), np.nanstd(etraj)))
        plt.tight_layout()
        flist.append(f3)
        
        
        frs = np.unique(df['frame'])
        efr = np.empty(len(frs))*np.nan
        efrstd = np.empty(len(frs))*np.nan
        for i in range(len(frs)):
            efr[i] = np.mean(df['Anisotropy'][df['frame']==frs[i]])
            efrstd[i] = np.std(df['Anisotropy'][df['frame']==frs[i]])
        f4 = plt.figure()
        plt.errorbar(frs, efr, efrstd, fmt='.')
        plt.ylabel('Avergae anisotropy on a frame')
        plt.xlabel('Frame')
        plt.ylim([-1,1])
        plt.tight_layout()
        flist.append(f4)
        
        
        # Plot the longest trajectories
        f5 = plt.figure()
        plt.xlabel('frame')
        plt.ylabel('Anisotropy')
        A = np.argsort(Ltraj)
        Nplot = 5 #number of plotted trajs        
        plt.title('Anisotropy of the %.0f longest trajectories'%(Nplot))
        inds = np.empty(Nplot, dtype=int)
        ind = 0
        while Nplot>0 and ind<len(trajs):
            #ind_th longest trajectory
            trajdat = df[df['particle']==trajs[A[-1-ind]]]
            if np.sum(trajdat['charge']==0.5)>=len(trajdat)/2: # if most of the defects are +1/2
                plt.plot(trajdat['frame'], trajdat['Anisotropy'], '-')
                Nplot -= 1
                inds[Nplot-1] = ind
            ind += 1
        plt.ylim([-1,1])
        plt.tight_layout()
        flist.append(f5)
        
        box_pts = 8
        box = np.ones(box_pts)/box_pts
        for j in range(len(inds)):
            trajdat = df[df['particle']==trajs[A[-1-inds[j]]]]
            f, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            ax1.plot(trajdat['frame'], np.convolve(trajdat['Anisotropy'], box, mode='same'), 'g-')
            ax2.plot(trajdat['frame'], np.convolve(trajdat['MinDist'], box, mode='same'), 'b-')
            ax1.set_xlabel('frame')
            ax1.set_ylabel('Anisotropy', color='g')
            ax2.set_ylabel('Distance to nearest neighbor', color='b')
            plt.title('Longest trajectory #%.0f (window av size %.0f)'%(j+1, box_pts))
            plt.tight_layout()
            
            flist.append(f)
        
    return flist
        
        

def check_tracking(imgpath, deftab_, track_param = [None, None, 0]):
    """
    From defect data (location, ...), perform defect tracking with parameter tuning.

    Parameters
    ----------
    imgpath : str
        Path to image from which detection is performed.
    deftab_ : Pandas DataFrame
        Contains defect information. We need location, charge, frame.
    track_param : size-3 array, optional
        Initial tracking parameters. The default is [None, None, 0].

    Returns deftab, track_param, [loopbutton, databutton, moviebutton, okbutton, startbutton]
    -------
    deftab : Pandas DataFrame
        Defect informations, to which the 'particle' column has been added or
        filled with trajectory id.
    track_param : size-3 iterable
        Contains selected tracking parameters
    refs : list of refecrences
        contains the references to sliders and buttons, for interactivity.

    """
    global quiver_artist
    global quiverM1
    global quiverM2
    global quiverM3
    global traj_artist
    global loop
    global deftab
    global deftab_raw
    
    searchR = track_param[0]
    memory = track_param[1]
    filt = track_param[2]
    
    loop = False
    if imgpath[-3:]=='tif':
        img_st = tf.imread(imgpath)
    else:
        img_st = plt.imread(imgpath)
    
    deftab_raw = deftab_
    deftab = deftab_raw#tp.filter_stubs(deftab_raw, filt)
    
    # if it is a multichannel image (color), take the first one 

    if img_st.ndim>3:
        img_st = img_st[:,0,:,:] #if we have several intensity channels take the first one
    img = img_st[0,:,:]

    #plt.figure()
    fig =  plt.figure()
    #plt.imshow(img, cmap='binary')
    
    #Initial slider value. /!\ DO NOT CORRESPOND NECESSARILY TO INITIAL TRACKING VALUES
    if searchR is None:
        if np.sum(np.logical_not(np.isnan(deftab['MinDist'])))>2:
            searchR = 4*np.nanmean(deftab['MinDist'])
        else:
            searchR = np.mean(img.shape)/4
    if memory is None:
        memory = max(round(len(np.unique(deftab['frame']))/15), 2)
    
    ani = [None]
    
    #sort the defects
    ptab = deftab[deftab['charge']==0.5]
    mtab = deftab[deftab['charge']==-0.5]
    otab = deftab[np.abs(deftab['charge'])!=0.5]
    
    deftab = pd.concat([ptab, mtab, otab])
    deftab = deftab.reset_index(drop=True)
    
    ############# Buttons  ##################
    # Start Animation #
    
    loopax = fig.add_axes([0.6, 0.2, 0.3, 0.07])
    loopbutton = CheckButtons(loopax, ["Loop movie"])
    
    startax = fig.add_axes([0.6, 0.375, 0.3, 0.07])
    startbutton = Button(startax, 'Preview video', hovercolor='0.975')
    dataax = fig.add_axes([0.6, 0.55, 0.3, 0.07])
    databutton = Button(dataax, 'Save Dataset', hovercolor='0.975')
    movieax = fig.add_axes([0.6, 0.725, 0.3, 0.07])
    moviebutton = Button(movieax, 'Save movie', hovercolor='0.975')
    okax = fig.add_axes([0.6, 0.9, 0.3, 0.07])
    okbutton = Button(okax, 'OK', hovercolor='0.975')
    
    
    def checkloop(event):
        global loop
        loop = not loop
    
    def Start_Animation(event):
        global quiver_artist
        global quiverM1
        global quiverM2
        global quiverM3
        global traj_artist
        global loop
        #plt.figure()
        figA, axA = plt.subplots()
        img_artist = axA.imshow(img_st[0,:,:], cmap='binary', animated=True)
        
        
        defframe = deftab[deftab['frame']==0]
        
        lim = 0.5
        e_map = 'PiYG'
        colorm = plt.get_cmap(e_map)
        defP = defframe[defframe['charge']==0.5]
        centroidsP = np.array([defP['y'], defP['x']]).transpose()
        axisP = np.array(defP['axis'])
        c = colorm(np.array(defP['Anisotropy'])/2/lim+0.5)
        quiver_artist = axA.quiver(centroidsP[:,1], centroidsP[:,0], np.cos(axisP), np.sin(axisP), angles='xy', color=c, edgecolor='k', linewidth=1)
        
        defM = defframe[defframe['charge']==-0.5]
        centroidsM = np.array([defM['y'], defM['x']]).transpose()
        axisM = np.array(defM['axis'])
        minuscolor = 'cornflowerblue'
        quiverM1 = axA.quiver(centroidsM[:,1], centroidsM[:,0], np.cos(axisM), np.sin(axisM), angles='xy', color=minuscolor)
        quiverM2 = axA.quiver(centroidsM[:,1], centroidsM[:,0], np.cos(axisM+2*np.pi/3), np.sin(axisM+2*np.pi/3), angles='xy', color=minuscolor)
        quiverM3 = axA.quiver(centroidsM[:,1], centroidsM[:,0], np.cos(axisM-2*np.pi/3), np.sin(axisM-2*np.pi/3), angles='xy', color=minuscolor)
        plt.colorbar(cm.ScalarMappable(norm=Normalize(-lim, lim), cmap=e_map), ax=axA, label='Splay-Bend Anisotropy []')
        
        if len(deftab):
            trajdata_x = [ [] for _ in range(int(np.max(deftab['particle'])+1)) ]
            trajdata_y = [ [] for _ in range(int(np.max(deftab['particle'])+1)) ]
            traj_artist = [None]*int(np.max(deftab['particle']+1))
        else:
            traj_artist = []
        #print(trajdata_x)
        defpart = np.array(np.unique(defframe['particle']), dtype=int)
        for i in range(len(defpart)):
            trajdata_x[defpart[i]].append(defframe['x'][defframe['particle']==defpart[i]].iloc[0])
            trajdata_y[defpart[i]].append(defframe['y'][defframe['particle']==defpart[i]].iloc[0])
        
        for i in range(len(traj_artist)):
            traj_artist[i], = axA.plot([], [])
            #print(trajdata_x[defpart[i]])
        
        #art_list = add_points(axA, deftab, 0, animated=True)
        ## create all frames
        #arts = []
        # for i in range(len(img_st)):
        #     im = axA.imshow(img_st[i,:,:], cmap='binary', animated=True)
        #     art_list = add_points(axA, deftab, i, animated=True)
        #     arts.append([im, *art_list])
            # arts_list = []
        
        def update(frame):
            global quiver_artist
            global quiverM1
            global quiverM2
            global quiverM3
            global traj_artist
            img_artist.set_array(img_st[frame,:,:])
            
            defframe = deftab[deftab['frame']==frame]
            
            quiver_artist.remove()
            quiverM1.remove()
            quiverM2.remove()
            quiverM3.remove()
            
            if frame==0:
                for i in range(len(traj_artist)):
                    trajdata_x[i] = []
                    trajdata_y[i] = []
                    traj_artist[i].set_data([], [])
            
            defP = defframe[defframe['charge']==0.5]
            centroidsP = np.array([defP['y'], defP['x']]).transpose()
            axisP = np.array(defP['axis'])
            c = colorm(np.array(defP['Anisotropy'])/2/lim+0.5)
            quiver_artist = axA.quiver(centroidsP[:,1], centroidsP[:,0], np.cos(axisP), np.sin(axisP), angles='xy', color=c, edgecolor='k', linewidth=1)
            
            defM = defframe[defframe['charge']==-0.5]
            centroidsM = np.array([defM['y'], defM['x']]).transpose()
            axisM = np.array(defM['axis'])
            minuscolor = 'cornflowerblue'
            quiverM1 = axA.quiver(centroidsM[:,1], centroidsM[:,0], np.cos(axisM), np.sin(axisM), angles='xy', color=minuscolor)
            quiverM2 = axA.quiver(centroidsM[:,1], centroidsM[:,0], np.cos(axisM+2*np.pi/3), np.sin(axisM+2*np.pi/3), angles='xy', color=minuscolor)
            quiverM3 = axA.quiver(centroidsM[:,1], centroidsM[:,0], np.cos(axisM-2*np.pi/3), np.sin(axisM-2*np.pi/3), angles='xy', color=minuscolor)
            
            defpart = np.array(np.unique(defframe['particle']), dtype=int)
            for i in range(len(defpart)):
                trajdata_x[defpart[i]].append(defframe['x'][defframe['particle']==defpart[i]].iloc[0])
                trajdata_y[defpart[i]].append(defframe['y'][defframe['particle']==defpart[i]].iloc[0])
                #print(trajdata_x[defpart[i]])
                if not (traj_artist[defpart[i]] is None):
                    traj_artist[defpart[i]].set_data(trajdata_x[defpart[i]], trajdata_y[defpart[i]])
        
            # if 'art_list' in globals():
            #     for j in range(len(art_list)):
            #         if art_list[j] is not None:
            #             art_list[j].remove()
            
            
            # art_list_temp = add_points(axA, deftab, frame, animated=True)
            # for j in range(max(len(art_list), len(art_list_temp))):
            #     if j<len(art_list):
            #         if j<len(art_list_temp):
            #             art_list[j] = art_list_temp[j]
            #         else:
            #             art_list[j] = None
            #     else:
            #         art_list.append(art_list_temp[j])
            
            #traj_artist[0].figure.canvas.draw_idle()
            return [img_artist, quiver_artist, quiverM1, quiverM2, quiverM3, *traj_artist]
        
        ## Start the animation
        #ani[0] = ArtistAnimation(figA, arts, interval=5, blit=False, repeat=loopbutton.get_active())
        ani[0] = FuncAnimation(figA, update, frames=range(len(img_st)), interval=5, blit=False, repeat=loop)#loopbutton.get_active())
        
        # while plt.fignum_exists(figA.number):
        #     plt.pause(0.1)
    
    startbutton.on_clicked(Start_Animation)
    
    def save_data(event):
        fold = filedialog.asksaveasfilename(defaultextension='.csv') # the user choses a place in file explorer
        #print(fold)
        if fold:
            deftab.to_csv(fold) # the DataFrame is saved as csv
            print('Data saved')
        else:
            print('Saving cancelled')
    
    def save_movie(event):
        fold = filedialog.asksaveasfilename(defaultextension='.tif') # the user choses a place in file explorer
        #writervideo = FFMpegWriter(fps=30)
        
        if fold:
            if ani[0] is None:
                Start_Animation(None)
                #plt.close()
            ani[0].save(fold, writer='pillow')#, fps=30)#, writer=writervideo)#, extra_args=['-vcodec', 'libx264']) # the DataFrame is saved as avi
            print('Movie Saved')
        else:
            print('Saving cancelled')
            
    is_open = True
    def finish(event):
        nonlocal is_open
        track_param[0] = sliders[1].val
        track_param[1] = sliders[0].val
        track_param[2] = sliders[2].val
        is_open = False
        plt.close(fig)
        #return deftab
        #return deftab, memslider.val, searchslider.val, filtslider.val, [loopbutton, databutton, moviebutton, okbutton, startbutton]
    
    loopbutton.on_clicked(checkloop)
    databutton.on_clicked(save_data)
    moviebutton.on_clicked(save_movie)
    okbutton.on_clicked(finish)
    
    
    ########## sliders #############
    slider_axes = []
    sliders     = []
    names       = ["Max skipped\n frames", "search\n range", "Filter small\n trajectories"]
    valmaxes    = [round(len(np.unique(deftab['frame']))/4), round(max(img.shape)/4), round(0.8*len(img_st))]
    inits       = [memory, searchR, filt]
    
    for i in range(len(inits)):
        axsl = fig.add_axes([0.1+0.2*i, 0.25, 0.0225, 0.63])
        thisslider = Slider(
            ax=axsl,
            label=names[i],
            valmin=1,
            valmax=valmaxes[i],
            valinit=inits[i],
            valstep=1,
            orientation="vertical"
        ) 
        slider_axes.append(axsl)
        sliders.append(thisslider)

    
    def change_tracking(val):
        global deftab
        global deftab_raw
        tp.quiet()
        
        #### Perform 3 tracking, for -1/2, +1/2 and others -> does not work
        ptab = deftab_raw[deftab_raw['charge']==0.5]
        mtab = deftab_raw[deftab_raw['charge']==-0.5]
        otab = deftab_raw[np.abs(deftab_raw['charge'])!=0.5]
        
        print(sliders[1].val)
        print(sliders[0].val)
        
        if len(ptab)>0:
            ptab = tp.link(ptab, search_range=sliders[1].val, memory=sliders[0].val)
        if len(mtab)>0:
            mtab = tp.link(mtab, search_range=sliders[1].val, memory=sliders[0].val)
        if len(otab)>0:
            otab = tp.link(otab, search_range=sliders[1].val, memory=sliders[0].val)

        deftab_raw = pd.concat([ptab, mtab, otab])
        
        # # prevent the particle number to be redundant
        ppart = ptab['particle'].to_numpy()
        mpart = mtab['particle'].to_numpy()
        opart = otab['particle'].to_numpy()
        
        # print(ppart)
        # print(mpart)
        #deftab['particle'] = [*ptab['particle'].to_numpy(), *mtab['particle'].to_numpy(), *otab['particle'].to_numpy()]

        
        mpart = mpart + np.max(ppart) + 1
        opart = opart + np.max(mpart) + 1
        
        
        deftab_raw['particle'] = [*ppart, *mpart, *opart]
        
        #### Trak defects irrespective of their charge. We can have mixed trajectories
        #tempdf = tp.link(deftab_raw, search_range=searchslider.val, memory=memslider.val)
        #deftab_raw['particle'] = tempdf['particle']
        if sliders[2].val:
            deftab_temp = tp.filter_stubs(deftab_raw, sliders[2].val)
            deftab = deftab_temp
        else:
            deftab = deftab_raw
        # Ndiff = len(deftab_temp) - len(deftab)
        # if Ndiff>0:    
        #     deftab.iloc[:,:] = deftab_temp.iloc[:len(deftab), :]
        #     for j in range(Ndiff):
        #         deftab.append(deftab_temp[len(deftab)+j])
        # else:
        #     deftab.drop(np.arange(len(deftab_temp),len(deftab)))
        #     deftab.iloc[:,:] = deftab_temp.iloc[:,:]
        
    
    for s in sliders:
        s.on_changed(change_tracking)
    
    
    while is_open:
        fig.canvas.flush_events()
        plt.pause(0.1)
    
    return deftab, track_param, [loopbutton, databutton, moviebutton, okbutton, startbutton]
        
    
def add_points(ax, all_data, frame, plot_cbar=False, animated=False):
    
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
    
    defect_df = all_data[all_data['frame']==frame]
    all_data = all_data[all_data['frame']<=frame]
    
    chargedef = np.array(defect_df['charge'])
    centroids = np.array([defect_df['y'], defect_df['x']]).transpose()
    es = np.array(defect_df['Anisotropy'])
    axisdef = np.array(defect_df['axis'])
    
    # arrows and annotations will be stored in artists_vec
    # length is Ndef + 2 artists per -1/2 defect, + 1 annotation per +1/2 + number of trajectories + colorbar
    artists_vec = [None]*(len(chargedef)+2*np.sum(np.abs(chargedef+0.5)<0.1)+np.sum(np.abs(chargedef-0.5)<0.1)+len(np.unique(all_data['particle']))+plot_cbar)
    
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
            incr += 2
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
        artists_vec[incr] = plt.colorbar(cm.ScalarMappable(norm=Normalize(-lim, lim), cmap=e_map), ax=ax, label='Splay-Bend Anisotropy []')
        incr += 1
    
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
    
    return artists_vec

def detect_defect_GUI(f_in=15, R_in=10, fname_in=None, frame_in=0):
    """
    Interface that allows to load an image and call the different other
    interfaces that performs detection etc.
    
    You can simply call:
    detect_defect_GUI()
    
    Parameters
    ----------
    f_in : numeral, optional
        Initial feature size to perform defect analysis. The default is 15.
    R_in : numeral, optional
        Initial detection radius to perform defect analysis. The default is 10.
    fname_in : string, optional
        Path to image to analyze. The default is None. Then the user can chose it.
    frame_in : int, optional
        Initial frame to analyze. The default is 0. Can be modified.

    Sliders
    -------
    frame_slider : 
        Choseframe that will serve to visualize the effect of the 
        parameters on the detection.
    
    Buttons
    -------
        load

    """
    global filename
    global defect_char
    global stack
    global img
    global track_param    
    global det_param
    vfield = None
    
    defect_char = pd.DataFrame()
    det_param = [f_in, R_in, 0.8]
    track_param = [4, 30, 0]
    stack = False
    unit = 'px'
    unit_per_px = 1
    unit_t = 'frame'
    unit_per_frame = 1
    
    keep = [None] # will store reference of interfaces
    
    filename = fname_in
    
    fig, ax = plt.subplots()
    
    # mng = plt.get_current_fig_manager()
    # mng.window.showMaximized()

    if not filename is None:
        img = tf.imread(filename)
        if filename[-3:]=='tif':
            with tf.TiffFile(filename) as tif:
                axes = tif.series[0].axes
                
                if "Z" in axes:
                    stack=True
                    if "C" in axes:
                        img = np.mean(img, axis=3)
                elif "C" in axes:
                    img = np.mean(img, axis=2)
                
                if tif.imagej_metadata:
                    try:
                        unit_maybe = tif.imagej_metadata['unit']
                        if unit_maybe!='':
                            unit=unit_maybe
                    except:
                        flipiti_useless_statement = 90
                    try:
                        unitt = tif.imagej_metadata['time unit']
                        unit_t = unitt
                        unit_per_frame = tif.imagej_metadata['finterval']
                    except:
                        unitt=1 # blank statement for the required except keyword
                xres = tif.pages[0].tags.get('XResolution')
                if xres:
                    xres = xres.value
                    unit_per_px = xres[1]/xres[0]
                
        else:
            img = plt.imread(filename)
            if len(img.shape)>2:
                stack = True

        ax.imshow(img, cmap='binary')
    else:
        img = plt.imread('..'+os.sep+'GUI_images'+os.sep+'spot_defect.jpg')
        ax.imshow(img, cmap='binary')
    plt.title('Image displayed for\n parameter choice')
    plt.subplots_adjust(bottom=0.2, left=0.4, right=0.8)  # Leave space for the button
    # buttons: load image // launch detection // check tracking // save // apply on other image
    # Display: frame_th image of laoded dataset
    
    loadax = fig.add_axes([0.05, 0.8, 0.25, 0.07])
    loadbutton = Button(loadax, 'Load', hovercolor='0.975')
    detax = fig.add_axes([0.05, 0.7, 0.25, 0.07])
    detbutton = Button(detax, 'Start Detection', hovercolor='0.975')
    trackax = fig.add_axes([0.05, 0.6, 0.25, 0.07])
    trackbutton = Button(trackax, 'Check tracking', hovercolor='0.975')
    saveax = fig.add_axes([0.05, 0.5, 0.25, 0.07])
    savebutton = Button(saveax, 'Save Data', hovercolor='0.975')
    dirax = fig.add_axes([0.05, 0.4, 0.25, 0.07])
    dirbutton = Button(dirax, 'Apply on\n directory', hovercolor='0.975')
    statax = fig.add_axes([0.05, 0.3, 0.25, 0.07])
    statbutton = Button(statax, 'Statistics', hovercolor='0.975')
     
    
    
    axframe = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    frame_slider = Slider(
        ax=axframe,
        label= 'Visualized\n frame',
        valmin=0,
        valmax=len(img),
        valinit=frame_in,
        valstep=1
    )
    
    def update_valmax(new_valmax):
        frame_slider.valmax = new_valmax  # Update the valmax attribute
        frame_slider._stop = new_valmax  # Update the internal _stop value
        frame_slider.ax.set_xlim(frame_slider.valmin, frame_slider.valmax)  # Update the slider's range
        frame_slider.val = min(frame_slider.val, new_valmax)  # Ensure the current value is within range
        frame_slider.set_val(frame_slider.val)  # Update the slider's value
        frame_slider.ax.figure.canvas.draw_idle()  # Redraw the slider
    
    def load_movie(event):
        global filename
        global stack
        global img
        global defect_char
        nonlocal vfield
        nonlocal unit
        nonlocal unit_per_px
        nonlocal unit_t
        nonlocal unit_per_frame
        fname = filedialog.askopenfilename()
        
        if fname: 
            if fname[-3:]=='tif':
                filename = fname
                img = tf.imread(filename)
                vfield = None
                with tf.TiffFile(filename) as tif:
                    axes = tif.series[0].axes
                    
                    if "Z" in axes or "T" in axes:
                        stack=True
                        if "C" in axes:
                            img = np.mean(img, axis=3)
                    elif "C" in axes:
                        img = np.mean(img, axis=2)
                    
                    if tif.imagej_metadata:
                        try:
                            unit_maybe = tif.imagej_metadata['unit']
                            if unit_maybe!='':
                                unit=unit_maybe
                                unitBox.set_val(unit)
                        except:
                            opla = 78
                            
                        try:
                            unitt = tif.imagej_metadata['time unit']
                            unit_t = unitt
                            unit_per_frame = tif.imagej_metadata['finterval']
                            unittBox.set_val(unit_t)
                            fpsBox.set_val(unit_per_frame)
                        except:
                            unitt=1 # blank statement for the required except keyword
                    xres = tif.pages[0].tags.get('XResolution')
                    if xres:
                        xres = xres.value
                        unit_per_px = xres[1]/xres[0]
                        uppxBox.set_val(unit_per_px)
                        
            elif fname[-3:]=='csv':
                defect_char = pd.read_csv(fname)
            elif fname[-3:]=='npy':
                vfield = np.load(fname)
            elif fname[-3]=='.mat':
                dat = scipy.io.loadmat(fname)
                x = dat['X']
                y = dat['Y']
                rho = dat['Rho']
                psi = dat['Psi']
            else:
                filename = fname
                img = plt.imread(filename)
                vfield = None
                if len(img.shape)>2:
                    stack = True
            if stack:
                ax.imshow(img[frame_slider.val,:,:], cmap='binary')
                update_valmax(len(img))
            else:
                ax.imshow(img, cmap='binary')
            fig.canvas.draw_idle()
    
        
    def detection(event):
        global defect_char
        global track_param
        nonlocal vfield
        global det_param
        if (filename is None) and (vfield is None):
            print('Load an image first!')
        else:
            defect_char, det_param, field, ref = defect_analyzer(filename, det_param, stack=stack, frame=frame_slider.val, um_per_px=unit_per_px, unit=unit, vfield=vfield, endsave=False)
            #vfield = field
            keep[0] = ref
        
        
            
    def check_track(event):
        global defect_char
        global track_param
        if len(defect_char)>0:
            defect_char, track_param, ref = check_tracking(filename, defect_char, track_param=track_param)#searchR=track_param[1], memory=track_param[0], filt=track_param[2])
            #print(track_param)
            keep[0] = ref
            #track_param = [track_param_0, track_param_0,track_param_0]
        else:
            print('You should perform detection first.')    
    
    def savedat(event):
        fold = filedialog.asksaveasfilename(defaultextension='.csv')
        
        if fold:
            defect_char_to_save = tp.filter_stubs(defect_char, track_param[2])
            
            #re-index particle column so that it is not absurd
            if 'particle' in defect_char_to_save.columns:
                part_vec = defect_char_to_save['particle'].to_numpy()
                part_list = np.unique(part_vec)
                for i in range(len(part_list)):
                    defect_char_to_save.loc[part_vec==part_list[i], 'particle']=i
            
            defect_char_to_save.to_csv(fold)
            print('Data Saved')
        else:
            print('Saving cancelled')
       
    
    def update_img(event):
        if stack:
            if not filename is None:
                ax.imshow(img[frame_slider.val,:,:], cmap='binary')
                fig.canvas.draw_idle()
    
    
    def on_directory(event):
        global det_param
        print('Apply the analysis with chosen parameters on a directory. Chose it!')
        print(det_param)
        folder = filedialog.askdirectory()
        bin_ = round(det_param[0]/4)
        sigma = round(1.5*det_param[0])
        #Loop over files
        for filename in os.listdir(folder):
            if filename.endswith('tif') or filename.endswith('png'):
                e_vec, err_vec, cost_vec, theta_vec, phi, defect_table = can.get_anisotropy(folder+os.sep+filename, False, det_param[1]/bin_, sigma, bin_, 2, 6, det_param[2], 0.75, plotit=False, stack=stack, savedir = None)
                
                defect_table.to_csv(folder+os.sep+'data_'+filename[:-3]+'csv')
                
                
                #plt.figure()
                fig, ax = plt.subplots()
                if filename.endswith('tif'):
                    imgtmp = tf.imread(folder+os.sep+filename)
                else:
                    imgtmp = plt.imread(folder+os.sep+filename)
                if stack:
                    # re arrange the trajectory list
                    defect_table.sort_values(by='frame')
                    order_traj = np.zeros(len(defect_table))
                    curr_part = defect_table['particle'].to_numpy()
                    old_list = np.unique(curr_part)
                    new_list = np.arange(len(old_list))                    
                    for i in range(len(old_list)):
                        storing_places = curr_part == old_list[i]
                        order_traj[storing_places] = new_list[i]
                    
                    imglist = []
                    for i in range(len(imgtmp)):
                        ax.imshow(imgtmp[i], cmap='gray')
                        add_points(ax, defect_table, i, plot_cbar=(not i))
                        fig.canvas.draw()
                        imgarray = np.copy(np.array(fig.canvas.renderer.buffer_rgba())[..., :3])
                        imglist.append(imgarray)
                        ax.clear()
                        # plt.close(fig)
                    tf.imwrite(folder+os.sep+'Traj_'+filename, np.stack(imglist, axis=0), photometric='rgb')
                    plt.close(fig)
                else:
                    plt.imshow(imgtmp, cmap='gray')
                    add_points(ax, defect_table, 0, plot_cbar=True)
                    fig.canvas.draw()
                    imgarray = np.array(fig.canvas.renderer.buffer_rgba())[..., :3]
                    plt.close(fig)
                    tf.imwrite(folder+os.sep+'Traj_'+filename, imgarray, photometric='rgb')
        print('Folder fully analyzed')
        
    def stat_func(event):
        global defect_char
        global img
        
        # nonlocal stack
        # nonlocal unit
        # nonlocal unit_per_px
        # nonlocal unit_t
        # nonlocal unit_per_frame
        # nonlocal vfield
        
        f = det_param[0]
        R = det_param[1]
        
        stat_me(defect_char, img=img, stack=stack, frame=0, unit=unit, unit_per_px=unit_per_px, tunit=unit_t, t_per_frame=unit_per_frame)
        ppattern, mpattern = defect_pattern(img, defect_char)
        orientation, coherence, ene, X, Y = OPy.orientation_analysis(ppattern, sigma=round(1.5*f), binning=round(f/4), plotf=False)
        phi, theta_unit = fan.compute_angle_diagram(orientation, R)
        e_pattern, err_e, costmin, theta_unit = can.one_defect_anisotropy(orientation, R=R)
        
        plt.figure()
        plt.imshow(ppattern, cmap='binary')
        plt.quiver(X, Y, np.cos(orientation), np.sin(orientation), angles='xy', scale=1/round(f/4), width=0.5, headaxislength=0, headlength=0, pivot='mid', color='red', units='xy')
        sh = ppattern.shape
        plt.plot(sh[0]/2+R*np.cos(phi), sh[1]/2+R*np.sin(phi))
        
        
        e_av_profile, average_theta = average_profile(defect_char, img, f, R)
        es, costs = fan.anisotropy_comparison(phi, average_theta)
        e_av_profile = es[np.argmin(costs)]
        
        plt.figure()
        plt.plot(phi, theta_unit, '.')
        plt.plot(phi, average_theta, '.')
        
        
        plt.figure()
        plt.imshow(ppattern, cmap='gray')
        plt.title ('Average field around +1/2 defect\n Anisotropy = %.2f\n Anisotropy (average profile) = %.2f\n Anisotropy (average value) = %.2f'%(e_pattern, e_av_profile, np.nanmean(defect_char['Anisotropy'])))
        plt.tight_layout()
        plt.figure()
        plt.imshow(mpattern, cmap='gray')
        plt.title ('Average field around -1/2 defect')
        if stack:
            fan.motility_analysis(defect_char, dt=1, unit_per_frame=unit_per_frame, unit_t = unit_t, unit_per_px = unit_per_px, unit_space = unit)
    
    textspaceax = fig.add_axes([0.85, 0.85, 0.12, 0.07])
    textspaceax.axis('off')
    textspaceax.text(0, 0, 'Space\nUnit')
    unitax = fig.add_axes([0.85, 0.65, 0.12, 0.07])
    unitBox = TextBox(unitax, '\n\n\n/px', initial=unit, label_pad=-0.5)
    uppxax = fig.add_axes([0.85, 0.75, 0.12, 0.07])
    uppxBox = TextBox(uppxax, '', initial=unit_per_px)
    texttimeax = fig.add_axes([0.85, 0.4, 0.12, 0.07])
    texttimeax.axis('off')
    texttimeax.text(0, 0, 'Time\nUnit')
    unittax = fig.add_axes([0.85, 0.2, 0.12, 0.07])
    unittBox = TextBox(unittax, '\n\n\n/frame', initial=unit_t, label_pad=-1)
    fpsax = fig.add_axes([0.85, 0.3, 0.12, 0.07])
    fpsBox = TextBox(fpsax, '', initial=unit_per_frame)
    
    def set_unit(text):
        nonlocal unit
        unit = text    
    def set_unit_per_px(text):
        nonlocal unit_per_px
        unit_per_px = float(text)
    def set_unit_t(text):
        nonlocal unit_t
        unit_t = text
    def set_unit_per_frame(text):
        nonlocal unit_per_frame
        unit_per_frame = float(text)
    
    unitBox.on_submit(set_unit)
    uppxBox.on_submit(set_unit_per_px)
    unittBox.on_submit(set_unit_t)
    fpsBox.on_submit(set_unit_per_frame)
    
    detbutton.on_clicked(detection)
    trackbutton.on_clicked(check_track)
    savebutton.on_clicked(savedat)
    loadbutton.on_clicked(load_movie)
    dirbutton.on_clicked(on_directory)
    frame_slider.on_changed(update_img)
    statbutton.on_clicked(stat_func)
    
    plt.show()
    # while plt.fignum_exists(fig.number):
    #     plt.pause(0.1)
    return [loadbutton, trackbutton, savebutton, detbutton, dirbutton, unitBox, uppxBox, unittBox, fpsBox, statbutton, keep]

def stat_me(dataset, img=None, stack=False, frame=0, unit='px', unit_per_px=1, tunit='frame', t_per_frame=1, min_dist=0):
    fset = []
    
    if img is None:
        area = 1
    elif stack:
        img0 = img[0]
        sh = img0.shape
        area = sh[0]*sh[1]*unit_per_px*unit_per_px
    else:
        sh = img.shape
        area = sh[0]*sh[1]*unit_per_px*unit_per_px
        
    
    if stack:
        frame_list = np.unique(dataset['frame'])
        Nplush = np.empty(len(frame_list))*np.nan
        Nminush = np.empty(len(frame_list))*np.nan
        Nplus = np.empty(len(frame_list))*np.nan
        Nminus = np.empty(len(frame_list))*np.nan
        N = np.empty(len(frame_list))*np.nan
        e_mean = np.empty(len(frame_list))*np.nan
        e_std = np.empty(len(frame_list))*np.nan
        
        for i in range(len(frame_list)):
            subset = dataset[dataset['frame']==frame_list[i]]
            Nplush[i] = np.sum(np.abs(subset['charge']-0.5)<0.25)
            Nminush[i] = np.sum(np.abs(subset['charge']+0.5)<0.25)
            Nplus[i] = np.sum(np.abs(subset['charge']-1)<0.25)
            Nminus[i] = np.sum(np.abs(subset['charge']+1)<0.25)
            N[i] = len(subset)
            subsubset = subset[subset['MinDist']<min_dist]
            e_mean[i] = np.nanmean(subsubset['Anisotropy'])
            e_std[i] = np.nanstd(subsubset['Anisotropy'])
        No = N - Nminus - Nminush - Nplush - Nplus
        
        # Density of defect over time
        fnum = plt.figure()
        if np.any(Nminus):
            plt.plot(frame_list*t_per_frame, Nminus/area, '.', label='-1 defect')
        if np.any(Nminush):
            plt.plot(frame_list*t_per_frame, Nminush/area, '.', label='-1/2 defect')
        if np.any(Nplush):
            plt.plot(frame_list*t_per_frame, Nplush/area, '.', label='+1/2 defect')
        if np.any(Nplus):
            plt.plot(frame_list*t_per_frame, Nplush/area, '.', label='+1 defect')
        if np.any(No):
            plt.plot(frame_list*t_per_frame, No/area, '.', label='other defects')
        plt.xlabel('Time ['+tunit+']')
        if img is None:
            plt.ylabel('Number of defects')
        else:
            plt.ylabel('Density of defects [1/'+unit+'$^2$]')
        plt.legend()
        plt.tight_layout()
        fset.append(fnum)
        
        # Mean anisotropy over time
        fte = plt.figure()
        plt.plot(frame_list*t_per_frame, e_mean)
        plt.fill_between(frame_list*t_per_frame, e_mean-e_std, e_mean+e_std, alpha=0.5, color=plt.gca().lines[-1].get_color())
        plt.xlabel('Time ['+tunit+']')
        plt.ylabel('Anisotropy')
        plt.tight_layout()
        fset.append(fte)
        
        # Defect movement
        
        
        # Defect density
        subset = dataset[dataset['frame']==frame]
        if not (img is None):
            img = img[frame]
        
    else:
        subset = dataset
    
    fdf = plt.figure()
    # density histogram
    if img is None:
        r = [[subset['x'].min(), subset['x'].max()], [subset['y'].min(), subset['y'].max()]]
        s = (r[0][1]-r[0][0], r[1][1]-r[1][0])
    else:
        plt.subplot(1,2,1)
        plt.imshow(img, cmap='gray')
        plt.plot(subset['x'], subset['y'], 'k.')
        plt.subplot(1,2,2)
        s = img.shape
        r = [[0, s[1]],[0, s[0]]]
    b = max(10, int(min(*s)/8)) # we want at least 8 points per box but at least 10 boxes
    
    
    # Density map at frame_th frame
    # X, Y = np.mgrid[r[0][0]:r[0][1]:b*1j, r[1][0]:r[1][1]:b*1j]
    if len(subset)>0:
        x_grid = np.linspace(r[0][0], r[0][1], b)
        y_grid = np.linspace(r[1][0], r[1][1], b)
        X, Y = np.meshgrid(x_grid, y_grid)
        dx = s[1] / (b - 1)  # pixel width
        dy = s[0] / (b - 1)  # pixel height
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([subset['x'], subset['y']])
        kernel = stats.gaussian_kde(values)
        density = np.reshape(kernel(positions).T, X.shape)/dx/dy
        Z = density * len(subset) / area / density.mean()
        plt.imshow(Z, cmap=plt.cm.gist_earth_r, extent=[*r[0], *r[1]], origin='lower')
    # plt.hist2d(subset['x'], subset['y'], bins=b, weights=np.ones(len(subset))*1/b/unit_per_px, range=r, cmap='Reds') #weights ensures unit consistency
    plt.plot(subset['x'], subset['y'], 'k.')
    plt.gca().invert_yaxis()
    plt.colorbar(label='Defect density [1/'+unit+'$^2$]')
    if stack:
        add_to_title = 'At t=%.0f'%(frame*t_per_frame)+tunit+'\n'
        fdf.suptitle(add_to_title + 'Average density: $%.1e\\pm %.1e$ 1/'%(np.mean(N/area), np.std(N/area)) + unit + '$^2$\n For +1/2: $%.1e\\pm %.1e$ 1/'%(np.mean(Nplush/area), np.std(Nplush/area)) + unit  + '$^2$\n For -1/2: $%.1e\\pm%.1e$ 1/'%(np.mean(Nminush/area), np.std(Nminush/area)) + unit + '$^2$')
    else:
        N = len(subset)
        Nplush = np.sum(subset['charge']==0.5)
        Nminush = np.sum(subset['charge']==-0.5)
        fdf.suptitle('Defect density: %.1e 1/'%(N/area) + unit + '$^2$\n For +1/2: %.1e 1/'%(Nplush/area) + unit  + '$^2$\n For -1/2: %.1e 1/'%(np.mean(Nminush/area)) + unit + '$^2$')
    plt.tight_layout()
    fset.append(fdf)
    
    fdhist = plt.figure()
    plt.hist(N/area, bins=20)
    plt.title('Average defect density: $%.1e\\pm %.1e$ 1/'%(np.mean(N/area), np.std(N/area))+unit+'$^2$')
    plt.xlabel('Defect density [1/'+unit+'$^2$]')
    plt.ylabel('Counts')
    plt.tight_layout()
    fset.append(fdhist)
    
    fh = plt.figure()
    subset = dataset[dataset['MinDist']>min_dist]
    plt.hist(subset['Anisotropy'], bins=20)
    plt.title('Average anisotropy: $%.2f\\pm%.2f$'%(np.nanmean(subset['Anisotropy']), np.nanstd(subset['Anisotropy'])))
    plt.xlim([-1,1])
    plt.xlabel('Anisotropy')
    plt.ylabel('Counts')
    plt.tight_layout()
    fset.append(fh)
    
    fdist = plt.figure()
    plt.hist(dataset['MinDist']*unit_per_px, bins=20)
    plt.plot([min_dist*unit_per_px, min_dist*unit_per_px], plt.ylim(), 'k--', label='Cut-off distance')
    plt.xlabel('Disatnce to nearest neighbor ['+unit+']')
    plt.ylabel('Counts')
    plt.legend()
    plt.title('Average: $%.2e\\pm%.2e$'%(np.nanmean(dataset['MinDist'])*unit_per_px, np.nanstd(dataset['MinDist'])*unit_per_px))
    plt.tight_layout()
    fset.append(fdist)
    
    if stack:
        color = dataset['frame']*t_per_frame#frame dependant 
    else:
        color = 'k'
    fdiste = plt.figure()
    plt.scatter(dataset['MinDist']*unit_per_px, dataset['Anisotropy'], marker='.', cmap=plt.cm.Wistia, c=color)
    if stack:
        plt.colorbar(label='Time ['+tunit+']')
    plt.plot(plt.xlim(), [0,0], 'k--')
    plt.plot([min_dist*unit_per_px, min_dist*unit_per_px], [-1,1], 'k:', label='Cut-off distance')
    plt.ylim([-1, 1])
    plt.xlabel('Distance to nearest neighbor ['+unit+']')
    plt.ylabel('Anisotropy')
    plt.tight_layout()
    fset.append(fdiste)
    
    
    return fset
        

def defect_pattern(field, dataset, cropsize = 100):    
    pset = dataset[np.abs(dataset['charge']-0.5)<0.2]
    mset = dataset[np.abs(dataset['charge']+0.5)<0.2]
    patterns_p = np.empty((len(pset), cropsize*2, cropsize*2))
    patterns_m = np.empty((len(mset), cropsize*2, cropsize*2))
    
    pincrement = 0
    mincrement = 0
    if 'particle' in dataset.columns: #if it is a movie
        part = np.unique(dataset['particle'])
        for i in range(len(field)):
            tpset = pset[pset['frame']==i]
            tmset = mset[mset['frame']==i]
            for ip in range(len(tpset)):
                xcrop, ycrop, rot_field = fan.crop_rotate_scalar(field[i], axis=-tpset['axis'].iloc[ip], cropsize=cropsize, xcenter=tpset['x'].iloc[ip], ycenter=tpset['y'].iloc[ip])
                patterns_p[pincrement] = rot_field
                pincrement +=1
                # plt.figure()
                # plt.imshow(rot_field, cmap='binary')
            for im in range(len(tmset)):
                xcrop, ycrop, rot_field = fan.crop_rotate_scalar(field[i], axis=-tmset['axis'].iloc[im], cropsize=cropsize, xcenter=tmset['x'].iloc[im], ycenter=tmset['y'].iloc[im])
                patterns_m[mincrement] = rot_field
                mincrement +=1
    else:

        for ip in range(len(pset)):
            xcrop, ycrop, rot_field = fan.crop_rotate_scalar(field, axis=-pset['axis'].iloc[ip], cropsize=cropsize, xcenter=pset['x'].iloc[ip], ycenter=pset['y'].iloc[ip])
            patterns_p[ip] = rot_field
        for im in range(len(mset)):
            xcrop, ycrop, rot_field = fan.crop_rotate_scalar(field, axis=-mset['axis'].iloc[im], cropsize=cropsize, xcenter=mset['x'].iloc[im], ycenter=mset['y'].iloc[im])
            patterns_m[im] = rot_field
    average_p = np.nanmean(patterns_p, axis=0)
    
    
    # for i in range(len(patterns_p)):
    #     plt.figure()
    #     plt.imshow(patterns_p[i], cmap='binary')
    
    return average_p, np.nanmean(patterns_m, axis=0)


def average_profile(defect_char, img, f, R):
    table = defect_char[defect_char['charge']==0.5]
    th_list = []
    ref = False
    for i in range(len(table)):
        orientation, coherence, ene, X, Y = OPy.orientation_analysis(img[table['frame'].iloc[i]], sigma=round(1.5*f), binning=round(f/4), plotf=False)
        # print(orientation.shape)
        # print(X.shape)
        # print(table['x'].iloc[i])
        # print(table['y'].iloc[i])
        e, err_e, costmin, theta_unit = can.one_defect_anisotropy(orientation, R=R, xc=table['x'].iloc[i]/2, yc=table['y'].iloc[i]/2, axis = table['axis'].iloc[i], plotit=ref)
        ref = False
        #phi, theta_unit = fan.compute_angle_diagram(orientation, R, center=(, ), axis=)
        th_list.append(theta_unit)
    
    theta = np.arctan2(np.nanmean(np.sin(th_list), axis=0), np.nanmean(np.cos(th_list), axis=0))
    return e, theta


%matplotlib qt
if __name__ == "__main__":
    keep = detect_defect_GUI()