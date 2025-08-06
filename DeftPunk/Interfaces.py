# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 18:28:44 2024
Last update Mon Aug 4 2025

@author: Quentin Chaboche

Interfaces.py

This module contains 3 functions creating interfaces to load an image, select
parameters for detection of defect location and anisotropy.

- detect_defect_GUI (general interface)
- defect_analyzer (defect detection interface)
- check_tracking (tracking interface)

Part of DeftPunk package
"""
import matplotlib.pyplot as plt
import numpy as np
import datetime
from matplotlib.widgets import Button, Slider, CheckButtons, TextBox
import pandas as pd
from DeftPunk import processing as pc
import DeftPunk.Analysis as an
import tifffile as tf
from matplotlib import cm
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
import tkinter
from tkinter import filedialog
import trackpy as tp
from matplotlib.animation import FuncAnimation
import scipy.io
import os
origin_file = os.path.abspath( os.path.dirname( __file__ ) )


bin_factor = 4

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



def load_image(imgpath, channel=0):
    if imgpath is None:
        return None, False, []
    
    if not os.path.exists(imgpath):
        raise FileNotFoundError(f"Image not found: {imgpath}")
    
    extension = imgpath.split('.')[-1]
    
    units = [1, 'px', 1, 'frame'] # default units
    stack = False
    if extension in ['tif', 'tiff']:
        # Read TIFF metadata using TiffFile
        with tf.TiffFile(str(imgpath)) as tif:
            axes = tif.series[0].axes  # E.g., "TZCYX"
            img = tif.asarray()
            # If T it's a time stack, if Z it's a z-stack
            stack = 'T' in axes or 'Z' in axes
            
            # determine if units are stored in the metadata
            # I use the try statement because if this fail the rest of the code works just fine
            if tif.imagej_metadata:
                try:
                    units[1] = tif.imagej_metadata['unit']
                except:
                    flipiti_useless_statement = 90 # Except cannot be empty
                try:
                    unitt = tif.imagej_metadata['time unit']
                    units[3] = unitt
                    units[2] = tif.imagej_metadata['finterval']
                except:
                    unitt=1 # blank statement for the required except keyword
            xres = tif.pages[0].tags.get('XResolution')
            if xres:
                xres = xres.value
                units[0] = xres[1]/xres[0]
    
    # handle other extension
    elif extension in ['gif', 'npz', 'npy']:
        raise NotImplementedError(f"{extension} format is not supported. Try tif, png, or a matplotlib-supported format.")            
    else: # png, jpg, bmp, automatically not stack
        img = plt.imread(imgpath)
        
    # reduce dimension if it is a multichannel image
    if len(img.shape)>2+stack:
        if channel==0: # 0 codes for averaging channels
            img = np.mean(img, axis=-1)
        else: # otherwise the channel can be selected
            img = img[::,channel]
    
    return img, stack, units


def defect_analyzer(imgpath, det_param, stack=True, frame=0, um_per_px=1, unit='px', vfield=None, endsave=True, savedir='Select'):
    """
    Calls the interface to analyze defect and their anisotropy on an image
       
    > defect_char, det_param, vfield, _ = defect_analyzer(imgpath, det_param, stack=True, frame=0, endsave=True, savedir='Select')
    
    The detection parameters are described at the end of the documentation.
    
    The function is structured as follow:
        - Input read-outs
        - Initialization
        - Sliders creation
        - Buttons creation
        - Exceptions and returns
    
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
    
    ############ Inupt read-out ###################
    # This one needs to be global so it is updated also in other interfaces like detect_defect_GUI
    # Because the two GUI execute in parallel
    global defect_char 
    
    w    = det_param[0]
    R    = det_param[1]
    over = False
    
    ### All necessary detection parameters. The 3 selected ones (f, R, o) define all others: ####
    sigma           = round(1.5*w) #integration size for orientation field
    bin_            = round(w/bin_factor) # DownSampling size for orientation field
    fsig            = 2 # in units of bin. Size of filter for nematic order parameter computation
    order_threshold = det_param[2]
    BoxSize         = 6
    peak_threshold  = 0.75
    
    # if the director field is an input, we lock this value
    if not (vfield is None):
        lock_field = True
    else:
        lock_field = False
     
    img_st, stack, _ = load_image(imgpath)
    if stack:
        img = img_st[frame]
    else:
        img = img_st
        
    field_visible = False # do we plot the field?   
    if img_st is None:
        field_visible = True
    
    ########## Initialization ######################
    
    ## Initial detection 
    # detection of defect location, axis and anisotropy
    e_vec, err_vec, cost_vec, theta_vec, phi, defect_char, vfield, pos = pc.get_anisotropy(img, False, R/bin_, sigma, bin_, fsig, BoxSize, order_threshold, peak_threshold, prescribed_field=vfield, plotit=False, stack=stack, savedir = None, give_field = True)
    fieldcolor  = 'navy'
    my_field    = [vfield, pos]
    
    fig, ax     = plt.subplots()
    
    #image 
    if not (img is None):
        back_img = plt.imshow(img, cmap='binary')
    # define director field display
    qline = ax.quiver(pos[0], pos[1], np.cos(vfield), np.sin(vfield), angles='xy', pivot='mid', headlength=0, headaxislength=0, scale_units='xy', scale=1/bin_ , color=fieldcolor)
    qline.set_visible(field_visible)
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    
    ## Initial display 
    # Lists of objects. It will be use to change their visibility and remove them
    art, R_vec = draw_defects(ax, defect_char, plot_cbar=True)
    
    # art_vec has vector field in index 0, then arrows and annotations
    art_vec = [qline, *art]
    
    ### Where we create the buttons and interface ###
    # adjust the main plot to make room for the sliders
    fig.subplots_adjust(left=0.22, bottom=0.20)

    # draw the legend
    axlegend = fig.add_axes([0.32, 0.87, 0.5, 0.13])
    imlegend = plt.imread('DeftPunk'+os.sep+'GUI_images'+os.sep+'defect_type.png')
    axlegend.imshow(imlegend)
    axlegend.axis('off')
    
    # draw the schematics of the defects with different anisotropy
    axschem = fig.add_axes([0.9, 0.19, 0.1, 0.7])
    imschem = plt.imread('DeftPunk'+os.sep+'GUI_images'+os.sep+'defect_style.png')
    axschem.imshow(imschem)
    axschem.axis('off')
    
    ############################# Sliders creation ##########################
    
    ## Creation of slider interactive objects
    # handle slider display. If physical units are provide display conversion
    if unit=='px':
        labw = 'Feature size [px]'
        labR = "Detection\n radius [px]"
    else:
        labw = 'Feature size [px] (%.2f '%(um_per_px*w)+unit+')'
        labR = "Detection radius [px]\n(%.2f "%(um_per_px*R)+unit+")"
        
    # Make a horizontal slider to control the feature size w.
    axw = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    w_slider    = Slider(
        ax      = axw,
        label   = labw,
        valmin  = 4,
        valmax  = 80,
        valinit = w,
        valstep = 1
    )
    
    # Make a vertically oriented slider to control the radius of detection R
    axR = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
    if img is None:
        Rmax = round(len(vfield)/4)
    else:
        Rmax = round(len(img)/4)
    R_slider = Slider(
        ax          = axR,
        label       = labR,
        valmin      = 2,
        valmax      = Rmax,
        valinit     = R,
        valstep     = 1,
        orientation = "vertical"
    )    
    
    # Make a vertically oriented slider to control the radius o fdetection R
    axThresh = fig.add_axes([0.23, 0.25, 0.0225, 0.63])
    Thresh_slider = Slider(
        ax = axThresh,
        label       = "Order parameter\n detection threshold",
        valmin      = 0,
        valmax      = 1,
        valinit     = order_threshold,
        valstep     = 0.05,
        orientation = "vertical"
        )
    
 
    
    ## Update functions for sliders 
    # The function to be called anytime a slider's value changes
    # Updating the feature_size w. It change director field, defect detection and anisotropies.    
    def update_w(val):
        global defect_char
        # values used eslewhere in the function
        nonlocal bin_
        nonlocal vfield
        nonlocal pos
        
        if not(unit=='px'):
            w_slider.label.set_text('Feature size [px] (%.2f '%(um_per_px*w_slider.val)+unit+')')
        
        # update parameter values
        sigma           = round(1.5*w_slider.val) #integration size for orientation field
        bin_            = round(w_slider.val/bin_factor) # Sampling size for orientation field
        fsig            = 2 # in units of bin. Size of filter for order parameter
        order_threshold = Thresh_slider.val
        BoxSize         = 6
        peak_threshold  = 0.75
        
        if lock_field:
            input_field = vfield
        else:
            input_field=None
            
        # re-detect defects
        e_vec, err_vec, cost_vec, theta_vec, phi, dchar, field, pos = pc.get_anisotropy(img, False, R_slider.val/bin_, sigma, bin_, fsig, BoxSize, order_threshold, peak_threshold, prescribed_field=input_field, plotit=False, stack=False, savedir = None, give_field=True)
        vfield = field
        defect_char = dchar
        my_field[0] = field
        my_field[1] = pos
        
        update_display(pos, fig, art_vec, R_vec, field, ax, R_slider.val, dchar, bin_)
    
    # update function for detection radius. Only anisotropy is changed
    def update_R(val):
        nonlocal vfield
        nonlocal pos
        
        # get useful parameter values
        field = my_field[0]
        bin_ = round(w_slider.val/bin_factor) # Sampling size for orientation field
        
        # to store the newly computed anisotropies 
        new_anisotropy = np.empty(len(defect_char))*np.nan
        
        if not(unit=='px'):
            R_slider.label.set_text("Detection radius [px]\n(%.2f "%(um_per_px*R_slider.val)+unit+")")
        
        # For every defect, re-compute anisotropy
        for i in range(len(defect_char)):
            e_vec_i, err_vec_i, cost_vec_i, th = pc.one_defect_anisotropy(field, R_slider.val/bin_, xc=defect_char['x'][i]/bin_, yc=defect_char['y'][i]/bin_, axis=defect_char['axis'][i])
            new_anisotropy[i] = e_vec_i
        defect_char['Anisotropy'] = new_anisotropy
        
        
        # update display
        update_display(pos, fig, art_vec, R_vec, vfield, ax, R_slider.val, defect_char, bin_)
        
    # update function for order_threshold. Defect detection is changed but not 
    # director field computation 
    def update_order(val):
        global defect_char
        nonlocal vfield
        
        # update parameter values
        sigma           = round(1.5*w_slider.val) #integration size for orientation field
        bin_            = round(w_slider.val/bin_factor) # Sampling size for orientation field
        fsig            = 2 # in units of bin. Size of filter for order parameter
        order_threshold = Thresh_slider.val
        BoxSize         = 6
        peak_threshold  = 0.75
        if lock_field:
            input_field = vfield
        else:
            input_field = None
            
        # re-perform detection (but not director field, since prescribed_field=input_field)
        e_vec, err_vec, cost_vec, theta_vec, phi, dchar, vfield, pos = pc.get_anisotropy(img, False, R_slider.val/bin_, sigma, bin_, fsig, BoxSize, order_threshold, peak_threshold, prescribed_field=input_field, plotit=False, stack=stack, savedir = None, give_field=True)
        defect_char = dchar
        my_field[0] = vfield
        my_field[1] = pos
        
        # get previous visibility info
        update_display(pos, fig, art_vec, R_vec, vfield, ax, R_slider.val, dchar, bin_)
    
    
    
    ## link slider object with update function 
    R_slider.on_changed(update_R)
    w_slider.on_changed(update_w)
    Thresh_slider.on_changed(update_order)
    
    
    ###################" Buttons creation #############################""
    
    ## helper to create a button
    def create_button(fig, x_pos, text, func):
        button_ax = fig.add_axes([x_pos, 0.025, 0.1, 0.04])
        new_button = Button(button_ax, text, hovercolor='0.975')
        new_button.on_clicked(func)
        return new_button
    

    ## update functions for buttons
    def reset(event):
        w_slider.reset()
        R_slider.reset()
        Thresh_slider.reset()
    
    def invert_color(event):
        if back_img.cmap(360)[0]: # This is True if cmap='gray' (cmap index 360 is (1,1,1) )
            back_img.set_cmap('binary')
        else:
            back_img.set_cmap('gray')
        fig.canvas.draw_idle()

    def finish(event):
        global defect_char
        nonlocal over
        
        det_param[0] = w_slider.val
        det_param[1] = R_slider.val
        det_param[2] = Thresh_slider.val
        
        class Placeholder: # just to be able to use fold.name line if savedir is input by the user
            def __init__(self, n):
                self.name = n
        
        if endsave:
            if savedir=='Select':
                print('Where to save the data?')
                root = tkinter.Tk()
                root.withdraw()  # Hide the empty main window
                root.call('wm', 'attributes', '.', '-topmost', '1') 
                # this function opens a browser and fold contains the saving path
                fold = filedialog.asksaveasfile(defaultextension='.csv') # the user choses a place in file explorer
                root.destroy()
            else:
                fold = Placeholder(savedir+os.sep+'data.csv')
        
        
        sigma = round(1.5*w_slider.val) #integration size for orientation field
        bin_ = round(w_slider.val/bin_factor) # Sampling size for orientation field
        
        over = True
        # If the image is a stack, perform detection for each frame
        if stack:
            print('Computing the whole stack...')
            e_vec, err_vec, cost_vec, theta_vec, phi, defect_char = pc.get_anisotropy(imgpath, False, R_slider.val/bin_, round(1.5*w_slider.val), round(w_slider.val/bin_factor), fsig, BoxSize, Thresh_slider.val, peak_threshold, plotit=False, stack=stack, savedir = None)
        plt.close(fig)
        
        if endsave:
            if not fold is None:
                # save the DataFrame as csv
                defect_char.to_csv(fold.name)
                # Write in a txt file the parameters
                paramfile = fold.name + '_parameters.txt'
                now_ = datetime.datetime.now()
                with open(paramfile, "a") as f:
                    f.write('At '+str(now_))
                    f.write('\nfeature size = %.0f '%(det_param[0]*um_per_px)+unit)
                    f.write('\nnematic order threshold = %.2f '%(det_param[2]))
                    f.write('\nDetection Radius = %.0f '%(det_param[1]*um_per_px)+unit)
                    
                print('Saved')
            else:
                print('Done')

            
    
    def plotField(event): # called by Director button 
        if art_vec[0].get_visible():
            art_vec[0].set_visible(False)
        else:
            art_vec[0].set_visible(True)
        fig.canvas.draw_idle()

    
    def plotR(event): # called by R-detection button
        # for each defect, change the visibiliy of the contour for R-detection
        for i in range(len(R_vec)):
            if not (R_vec[i] is None):
                if R_vec[i].get_visible():
                    R_vec[i].set_visible(False)
                else:
                    R_vec[i].set_visible(True)
        fig.canvas.draw_idle()
        
        # adjust the frame 
        new_xlim = ax.get_xlim()
        new_ylim = ax.get_ylim()
        if new_xlim[0]<x_lim[0] or new_xlim[1]>x_lim[1]: 
            ax.set_xlim(x_lim)
        if new_ylim[0]>y_lim[0] or new_ylim[1]<y_lim[1]: 
            ax.set_ylim(y_lim)

    
    def ClickSave(event): # called by the save button
        # create another figure
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
        draw_defects(axsave, defect_char, plot_cbar=True, R_vis=R_vis)
        
        # set the main figure display range
        axsave.set_xlim(ax.get_xlim())
        axsave.set_ylim(ax.get_ylim())
        # Write parameters as title
        axsave.set_title('feature size = %.0f px, R = %.0f px\norder threshold = %.2f'%(w_slider.val, R_slider.val, Thresh_slider.val))
        root = tkinter.Tk()
        root.withdraw()  # Hide the empty main window
        root.call('wm', 'attributes', '.', '-topmost', '1') 
        fold = filedialog.asksaveasfile(defaultextension='.png') # make the user choose a file location
        root.destroy()
        if fold:
            figsave.savefig(fold.name) # save figure at this location
            print('Saved')
        else:
            print('Saving cancelled')
        plt.close(figsave)
      
    ## create the buttons
    reversebutton = create_button(fig, 0.05, 'Invert Color', invert_color)
    resetbutton   = create_button(fig, 0.5, 'Reset', reset)
    OKbutton      = create_button(fig, 0.8, 'OK', finish)
    Fieldbutton   = create_button(fig, 0.2, 'Director', plotField)
    Circlebutton  = create_button(fig, 0.35, 'R-Detection', plotR)
    Savebutton    = create_button(fig, 0.65, 'Save Image', ClickSave)
    
    
    ########################### End of function ##########################
    
    # Throw exception if the figure is closed by hand
    def on_close(event):
        if not over:
            raise Exception("Program interrupted by the user (closed the figure).") 
    fig.canvas.mpl_connect('close_event', on_close)
    
    ### In case there is a problem with synchrony you can uncomment this (the execution will be stopped until the figure is closed)
    # 
    # while plt.fignum_exists(fig.number):
    #     plt.pause(0.1)
        
    return defect_char, det_param, vfield, [OKbutton, Savebutton, Circlebutton, Fieldbutton, resetbutton, reversebutton]
        
def check_tracking(imgpath, deftab_, track_param = [None, None, 0]):
    """
    From defect data (location, frame,..), perform defect tracking with parameter tuning.

    Parameters
    ----------
    imgpath : str
        Path to image from which detection is performed.
    deftab_ : Pandas DataFrame
        Contains defect information. We need location, charge, frame.
    track_param : size-3 array, optional
        Initial tracking parameters. The default is [None, None, 0].

    Returns 
    -------
    deftab : Pandas DataFrame
        Defect informations, to which the 'particle' column has been added or
        filled with trajectory id.
    track_param : size-3 iterable
        Contains selected tracking parameters
    refs : list of refecrences
        contains the references to sliders and buttons, for interactivity.

    """
    # objects to be updated and kept along the function execution
    global quiver_artist
    global quiverM1
    global quiverM2
    global quiverM3
    global traj_artist
    global loop
    global defect_char
    global deftab_raw
    
    # get the parameters
    searchR = track_param[0]
    memory = track_param[1]
    filt = track_param[2]
    loop = False
    
    # load the stack
    if imgpath[-3:]=='tif':
        img_st = tf.imread(imgpath)
    else:
        img_st = plt.imread(imgpath)
    
    # create temporary tables to modify the files
    deftab_raw = deftab_
    defect_char = deftab_raw
    
    # if it is a multichannel image (color), take the first channel 
    if img_st.ndim>3:
        img_st = img_st[:,0,:,:] #if we have several intensity channels take the first one
    img = img_st[0,:,:]

    fig =  plt.figure()
    
    #Initial slider value. /!\ DO NOT CORRESPOND NECESSARILY TO INITIAL TRACKING VALUES
    if searchR is None:
        if np.sum(np.logical_not(np.isnan(defect_char['MinDist'])))>2:
            searchR = 4*np.nanmean(defect_char['MinDist'])
        else:
            searchR = np.mean(img.shape)/4
    if memory is None:
        memory = max(round(len(np.unique(defect_char['frame']))/15), 2)
    
    # This will contain the animation (in a list so it is modified inside functions)
    ani = [None]
    
    #sort the defects according to charge
    ptab = defect_char[defect_char['charge']==0.5]
    mtab = defect_char[defect_char['charge']==-0.5]
    otab = defect_char[np.abs(defect_char['charge'])!=0.5]
    
    defect_char = pd.concat([ptab, mtab, otab])
    defect_char = defect_char.reset_index(drop=True)
    
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
        # start with drawing 1st frame
        img_artist = axA.imshow(img_st[0,:,:], cmap='binary', animated=True)
        
        # take info from frame 0
        defframe = defect_char[defect_char['frame']==0]
        
        # for plus and minus defects, get informations from table
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
        
        # initialize lists for trajectory plotting
        if len(defect_char):
            trajdata_x = [ [] for _ in range(int(np.max(defect_char['particle'])+1)) ]
            trajdata_y = [ [] for _ in range(int(np.max(defect_char['particle'])+1)) ]
            traj_artist = [None]*int(np.max(defect_char['particle']+1))
        else:
            traj_artist = []
            
        defpart = np.array(np.unique(defframe['particle']), dtype=int)
        # append to the list from the data of the frame 0 
        for i in range(len(defpart)):
            trajdata_x[defpart[i]].append(defframe['x'][defframe['particle']==defpart[i]].iloc[0])
            trajdata_y[defpart[i]].append(defframe['y'][defframe['particle']==defpart[i]].iloc[0])
        
        # prepare the figure
        for i in range(len(traj_artist)):
            traj_artist[i], = axA.plot([], [])

        
        def update(frame): # for this animation, the next frame is updated from current frame
            global quiver_artist
            global quiverM1
            global quiverM2
            global quiverM3
            global traj_artist
            
            # current image is the frame number `frame`
            img_artist.set_array(img_st[frame,:,:])
            
            # points from current frame
            defframe = defect_char[defect_char['frame']==frame]
            
            # clean objects from previous update
            quiver_artist.remove()
            quiverM1.remove()
            quiverM2.remove()
            quiverM3.remove()
            
            if frame==0:
                for i in range(len(traj_artist)):
                    trajdata_x[i] = []
                    trajdata_y[i] = []
                    traj_artist[i].set_data([], [])
            
            # split plus and minus defects
            defP = defframe[defframe['charge']==0.5]
            centroidsP = np.array([defP['y'], defP['x']]).transpose()
            axisP = np.array(defP['axis'])
            c = colorm(np.array(defP['Anisotropy'])/2/lim+0.5)
            # plot +1/2 defects 
            quiver_artist = axA.quiver(centroidsP[:,1], centroidsP[:,0], np.cos(axisP), np.sin(axisP), angles='xy', color=c, edgecolor='k', linewidth=1)
            
            defM = defframe[defframe['charge']==-0.5]
            centroidsM = np.array([defM['y'], defM['x']]).transpose()
            axisM = np.array(defM['axis'])
            minuscolor = 'cornflowerblue'
            # plot -1/2 defects
            quiverM1 = axA.quiver(centroidsM[:,1], centroidsM[:,0], np.cos(axisM), np.sin(axisM), angles='xy', color=minuscolor)
            quiverM2 = axA.quiver(centroidsM[:,1], centroidsM[:,0], np.cos(axisM+2*np.pi/3), np.sin(axisM+2*np.pi/3), angles='xy', color=minuscolor)
            quiverM3 = axA.quiver(centroidsM[:,1], centroidsM[:,0], np.cos(axisM-2*np.pi/3), np.sin(axisM-2*np.pi/3), angles='xy', color=minuscolor)
            
            defpart = np.array(np.unique(defframe['particle']), dtype=int)
            # append the trajectory data with current defects.
            for i in range(len(defpart)):
                trajdata_x[defpart[i]].append(defframe['x'][defframe['particle']==defpart[i]].iloc[0])
                trajdata_y[defpart[i]].append(defframe['y'][defframe['particle']==defpart[i]].iloc[0])
                if not (traj_artist[defpart[i]] is None):
                    traj_artist[defpart[i]].set_data(trajdata_x[defpart[i]], trajdata_y[defpart[i]])

            return [img_artist, quiver_artist, quiverM1, quiverM2, quiverM3, *traj_artist]
        
        ## Start the animation
        # FuncAnimation make an animation from a figure and an update function
        ani[0] = FuncAnimation(figA, update, frames=range(len(img_st)), interval=5, blit=False, repeat=loop)#loopbutton.get_active())
        
        # while plt.fignum_exists(figA.number):
        #     plt.pause(0.1)
    
    startbutton.on_clicked(Start_Animation)
    
    def save_data(event):
        # call browser for user to select save path
        root = tkinter.Tk()
        root.withdraw()
        root.call('wm', 'attributes', '.', '-topmost', '1')  # Bring dialog to front (optional)
        fold = filedialog.asksaveasfilename(defaultextension='.csv') # the user choses a place in file explorer
        root.destroy()
        
        if fold:
            defect_char.to_csv(fold) # the DataFrame is saved as csv
            # parameters are stored in a txt file
            paramfile = fold[:-4] + '_parameters.txt'
            now_ = datetime.datetime.now()
            with open(paramfile, "a") as f:
                f.write('At '+str(now_))
                f.write('\nsearch range = %.0f '%(track_param[0])+' px')
                f.write('\nmemory = %.0f '%(track_param[1]) + ' frames')
                f.write('\nfilter (minimum trajectory length) = %.0f '%(track_param[2])+' frames')
            print('Data saved')
        else:
            print('Saving cancelled')
    
    def save_movie(event):
        # call a browser
        root = tkinter.Tk()
        root.withdraw()
        root.call('wm', 'attributes', '.', '-topmost', '1')  # Bring dialog to front (optional)
        fold = filedialog.asksaveasfilename(defaultextension='.tif') # the user choses a place in file explorer
        root.destroy()
        
        if fold: # is the user selected a name
            if ani[0] is None:
                Start_Animation(None) # create animation
            ani[0].save(fold, writer='pillow')
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
    
    loopbutton.on_clicked(checkloop)
    databutton.on_clicked(save_data)
    moviebutton.on_clicked(save_movie)
    okbutton.on_clicked(finish)
    
    
    ########## sliders #############
    slider_axes = []
    sliders     = []
    names       = ["Max skipped\n frames", "search\n range", "Filter small\n trajectories"]
    valmaxes    = [round(len(np.unique(defect_char['frame']))/4), round(max(img.shape)/4), round(0.8*len(img_st))]
    inits       = [memory, searchR, filt]
    
    # iteratively create sliders (same update function)
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

    
    def change_tracking(val): # update function for the 3 sliders
        global defect_char
        global deftab_raw
        tp.quiet()
        
        #### Perform 3 tracking, for -1/2, +1/2 and others
        ptab = deftab_raw[deftab_raw['charge']==0.5]
        mtab = deftab_raw[deftab_raw['charge']==-0.5]
        otab = deftab_raw[np.abs(deftab_raw['charge'])!=0.5]
        
        
        if len(ptab)>0:
            ptab = tp.link(ptab, search_range=sliders[1].val, memory=sliders[0].val)
        if len(mtab)>0:
            mtab = tp.link(mtab, search_range=sliders[1].val, memory=sliders[0].val)
        if len(otab)>0:
            otab = tp.link(otab, search_range=sliders[1].val, memory=sliders[0].val)

        deftab_raw = pd.concat([ptab, mtab, otab])
        
        # # prevent the particle number to be redundant when tables are merged again
        ppart = ptab['particle'].to_numpy()
        mpart = mtab['particle'].to_numpy()
        opart = otab['particle'].to_numpy()
        
        mpart = mpart + np.max(ppart) + 1
        opart = opart + np.max(mpart) + 1
        
        # merge table from the -1/2, +1/2 and others
        deftab_raw['particle'] = [*ppart, *mpart, *opart]
        
        # filter small trajectories
        if sliders[2].val:
            deftab_temp = tp.filter_stubs(deftab_raw, sliders[2].val)
            defect_char = deftab_temp
        else:
            defect_char = deftab_raw
        
    # iteratively activate sliders
    for s in sliders:
        s.on_changed(change_tracking)
    
    # This stops execution until detection is finished.
    # It makes the function blink but I need it otherwise return deftab is not correct
    # (it return the tracking with initial parameters and not chosen ones)
    # Working on a solution
    # while is_open:
    #     fig.canvas.flush_events()
    #     plt.pause(0.1)
    
    return defect_char, track_param, [loopbutton, databutton, moviebutton, okbutton, startbutton]
            
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

def detect_defect_GUI(f_in=15, R_in=10, fname_in=None, frame_in=0):
    """
    Interface that allows to load an image and call the different other
    interfaces that performs detection etc.
    
    You can simply call:
    _ = detect_defect_GUI()
    
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
    
    Returns
    ---------
    refs : list
        references to the buttons (for interactivity)

    Sliders
    -------
    frame_slider : 
        Choose the frame that will serve to visualize the effect of the 
        parameters on the detection.
    
    Buttons
    -------
    load : 
        Open a file explorer that loads an image
        Alternatively, you can load a defect-detection dataset, or a director field
    Start Detection : 
        Open the defect detection interface with selected image at selected frame
    Check Tracking :
        Open the tracking interface with selected stack and performed detection
    Save Data :
        Saves the detection data as a .csv in a chosen location (from file explorer)
    Apply on Directory :
        Applies the detection with chosen parameters on a selected directory
    Statistics :
        Applies a set a basic statistics. See documentation for stat_me.

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
        img, stack, _ = load_image(filename)
        ax.imshow(img, cmap='binary')
    else:
        img = plt.imread('DeftPunk'+os.sep+'GUI_images'+os.sep+'spot_defect.jpg')
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
        root = tkinter.Tk()
        root.withdraw()
        root.call('wm', 'attributes', '.', '-topmost', '1')  # Bring dialog to front (optional)
        fname = filedialog.askopenfilename()
        root.destroy()
        
        if fname: 
            extension = fname.split('.')[-1]
            
            if extension=='csv':
                defect_char = pd.read_csv(fname)
            elif extension=='npy':
                vfield = np.load(fname)
            elif extension=='.mat':
                dat = scipy.io.loadmat(fname)
                x = dat['X']
                y = dat['Y']
                rho = dat['Rho']
                psi = dat['Psi']
            else:
                img, stack, units = load_image(fname)
                filename = fname
                vfield   = None
                
                if units[1]!='':
                    unitBox.set_val(units[1])
                unittBox.set_val(units[3])
                fpsBox.set_val(units[2])
                uppxBox.set_val(units[0])
                
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
        root = tkinter.Tk()
        root.withdraw()
        root.call('wm', 'attributes', '.', '-topmost', '1')  # Bring dialog to front (optional)
        fold = filedialog.asksaveasfilename(defaultextension='.csv')
        root.destroy()
        
        if fold:
            
            #re-index particle column so that it is not absurd
            if 'particle' in defect_char.columns:
                defect_char_to_save = tp.filter_stubs(defect_char, track_param[2])
                part_vec = defect_char_to_save['particle'].to_numpy()
                part_list = np.unique(part_vec)
                for i in range(len(part_list)):
                    defect_char_to_save.loc[part_vec==part_list[i], 'particle']=i
            else:
                
                defect_char_to_save = defect_char
            
            defect_char_to_save.to_csv(fold)
            
            paramfile = fold[:-4] + '_parameters.txt'
            now_ = datetime.datetime.now()
            with open(paramfile, "a") as f:
                f.write('At '+str(now_))
                
                f.write('\nfeature size = %.0f '%(det_param[0]*unit_per_px)+unit)
                f.write('\nnematic order threshold = %.2f '%(det_param[2]))
                f.write('\nDetection Radius = %.0f '%(det_param[1]*unit_per_px)+unit)
                f.write('\nsearch range = %.0f '%(track_param[0]*unit_per_px)+unit)
                f.write('\nmemory = %.0f '%(track_param[1]*unit_per_frame) + unit_t)
                f.write('\nfilter (minimum trajectory length) = %.0f '%(track_param[2]*unit_per_frame)+unit_t)
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
        root = tkinter.Tk()
        root.withdraw()
        root.call('wm', 'attributes', '.', '-topmost', '1')  # Bring dialog to front (optional)
        folder = filedialog.askdirectory()
        root.destroy()
        bin_ = round(det_param[0]/4)
        sigma = round(1.5*det_param[0])
        #Loop over files
        for filename in os.listdir(folder):
            if filename.endswith('tif') or filename.endswith('png'):
                e_vec, err_vec, cost_vec, theta_vec, phi, defect_table = pc.get_anisotropy(folder+os.sep+filename, False, det_param[1]/bin_, sigma, bin_, 2, 6, det_param[2], 0.75, plotit=False, stack=stack, savedir = None)
                
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
                        draw_defects(ax, defect_table, i, plot_cbar=(not i))
                        fig.canvas.draw()
                        imgarray = np.copy(np.array(fig.canvas.renderer.buffer_rgba())[..., :3])
                        imglist.append(imgarray)
                        ax.clear()
                        # plt.close(fig)
                    tf.imwrite(folder+os.sep+'Traj_'+filename, np.stack(imglist, axis=0), photometric='rgb')
                    plt.close(fig)
                else:
                    plt.imshow(imgtmp, cmap='gray')
                    draw_defects(ax, defect_table, 0, plot_cbar=True)
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
        
        an.stat_me(defect_char, img=img, stack=stack, frame=0, unit=unit, unit_per_px=unit_per_px, tunit=unit_t, t_per_frame=unit_per_frame)
        ppattern, mpattern = an.defect_pattern(img, defect_char)
        orientation, coherence, ene, X, Y = pc.orientation_analysis(ppattern, sigma=round(1.5*f), binning=round(f/4), plotf=False)
        phi, theta_unit = pc.compute_angle_diagram(orientation, R)
        e_pattern, err_e, costmin, theta_unit = pc.one_defect_anisotropy(orientation, R=R)
        
        plt.figure()
        plt.imshow(ppattern, cmap='binary')
        plt.quiver(X, Y, np.cos(orientation), np.sin(orientation), angles='xy', scale=1/round(f/4), width=0.5, headaxislength=0, headlength=0, pivot='mid', color='red', units='xy')
        sh = ppattern.shape
        plt.plot(sh[0]/2+R*np.cos(phi), sh[1]/2+R*np.sin(phi))
        
        
        e_av_profile, average_theta = pc.average_profile(defect_char, img, f, R)
        es, costs = pc.anisotropy_comparison(phi, average_theta)
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
            an.motility_analysis(defect_char, dt=1, unit_per_frame=unit_per_frame, unit_t = unit_t, unit_per_px = unit_per_px, unit_space = unit)
    
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
        


# if __name__ == "__main__":
#     keep = detect_defect_GUI()