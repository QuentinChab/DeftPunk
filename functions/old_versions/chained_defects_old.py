# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 18:28:44 2024

@author: Quentin

This script contains only one functions: defect_analyzer.
It's the highest function of the hierarchy: it only treats interface
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
import pandas as pd
import compute_anisotropy as can
import anisotropy_functions as fan
import tifffile as tf
from matplotlib import cm
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
from tkinter import filedialog
import os

origin_file = os.path.abspath( os.path.dirname( __file__ ) )

bin_factor = 4

def defect_analyzer(imgpath, w, R, stack=True, frame=0):
    """Calls the interface to analyze defect and their anisotropy on an image
       
    The exact choice of detection parameter is described at the end.
    
    Parameters
       ----------
       imgpath : str
           path to the image to analyze.
       w : int or float
           feature size in pixel on which most of detection parameters are chained to.
           As an example, the window size for vector field computation is sigma=1.5*w
           The other relations are described at the end.
       R : int or float
           Radius of detection. Around a defect a contour is taken at distance R
           and the director field is taken on this contour. This is used to 
           compute defect anisotropy.
       stack : bool, optional
           If the image is a stack or not.
           Default is True
       frame : int, optional
           If the image is a stack, index of the frame displayed at the interface
           Default is 0
           
           
    Returns
       ----------
       defect_char : Pandas DataFrame
           Contains all the informations of detected defects. The fields are
           'charge', 'axis', 'x', 'y', 'Anisotropy' and 'Error'.
           If the image is a stack we also have the fields
           'frame', 'MinDist', 'particle'
           MinDist is the distance to the closest other defect from the same frame
           Particle is an identifier for defect tracking. 
       
       w_out : float
           Value for feature size w chosen with the slider.
       R_out : float
           Value for detection radius R chosen with the slider.
       order_out : float
           Value for order_threshold parameter chosen with the slider.
           
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
    ### Where we define the relation between the parameters ####
    sigma = round(1.5*w) #integration size for orientation field
    bin_ = round(w/bin_factor) # Sampling size for orientation field
    fsig = 2 # in units of bin. Size of filter for order parameter
    order_threshold = 0.4*fsig
    BoxSize = 6
    peak_threshold = 0.75
    
    ### Where we load the image and select the displayed frame 'frame'
    #use the right unpacking package
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
    e_vec, err_vec, cost_vec, theta_vec, phi, defect_char, vfield, pos = can.get_anisotropy(img, False, R/bin_, sigma, bin_, fsig, BoxSize, order_threshold, peak_threshold, plotit=False, stack=stack, savedir = None, give_field = True)
    fieldcolor = 'navy'
    my_field = [vfield, pos]
    
    fig, ax = plt.subplots()
    #image 
    plt.imshow(img, cmap='binary')
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    # vector field
    qline = ax.quiver(pos[0], pos[1], np.cos(vfield), np.sin(vfield), angles='xy', pivot='mid', headlength=0, headaxislength=0, scale_units='xy', scale=1/bin_ , color=fieldcolor)
    qline.set_visible(False)
    e_map = 'PiYG'
    colorm = cm.get_cmap(e_map)
    
    lim = 0.5 # limits of anisotropy colorbar
    
    ##### Sliders ############
    # Make a horizontal slider to control the feature size w.
    axw = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    w_slider = Slider(
        ax=axw,
        label='Feature size [px]',
        valmin=4,
        valmax=80,
        valinit=w,
        valstep=1
    )
    
    # Make a vertically oriented slider to control the radius o fdetection R
    axR = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
    R_slider = Slider(
        ax=axR,
        label="Detection radius [px]",
        valmin=2,
        valmax=round(len(img)/4),
        valinit=R,
        valstep=1,
        orientation="vertical"
    )    
    
    # Make a vertically oriented slider to control the radius o fdetection R
    axThresh = fig.add_axes([0.2, 0.25, 0.0225, 0.63])
    Thresh_slider = Slider(
        ax=axThresh,
        label="Threshold",
        valmin=0,
        valmax=1,
        valinit=order_threshold,
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
    imlegend = plt.imread(origin_file+'/GUI_images/defect_type.png')
    axlegend.imshow(imlegend)
    axlegend.axis('off')
    
    # draw the schematics of the defects with different anisotropy
    axschem= fig.add_axes([0.9, 0.19, 0.1, 0.7])
    imschem = plt.imread(origin_file+'/GUI_images/defect_style.png')
    axschem.imshow(imschem)
    axschem.axis('off')
    
    ##### Update functions for sliders
    # The function to be called anytime a slider's value changes
    def update_w(val):
        global defect_char
        global bin_
        
        # get new defects and anisotropies
        sigma = round(1.5*w_slider.val) #integration size for orientation field
        bin_ = round(w_slider.val/bin_factor) # Sampling size for orientation field
        fsig = 2 # in units of bin. Size of filter for order parameter
        order_threshold = Thresh_slider.val
        BoxSize = 6
        peak_threshold = 0.75
        e_vec, err_vec, cost_vec, theta_vec, phi, dchar, field, pos = can.get_anisotropy(img, False, R_slider.val/bin_, sigma, bin_, fsig, BoxSize, order_threshold, peak_threshold, plotit=False, stack=stack, savedir = None, give_field=True)
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
        sigma = round(1.5*w_slider.val) #integration size for orientation field
        bin_ = round(w_slider.val/bin_factor) # Sampling size for orientation field
        fsig = 2 # in units of bin. Size of filter for order parameter
        order_threshold = Thresh_slider.val
        BoxSize = 6
        peak_threshold = 0.75
        e_vec, err_vec, cost_vec, theta_vec, phi, dchar, vfield, pos = can.get_anisotropy(img, False, R_slider.val/bin_, sigma, bin_, fsig, BoxSize, order_threshold, peak_threshold, plotit=False, stack=stack, savedir = None, give_field=True)
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

    # Create 5 `matplotlib.widgets.Button` 
    resetax = fig.add_axes([0.5, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')
    OKax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    OKbutton = Button(OKax, 'OK', hovercolor='0.975')
    Fieldax = fig.add_axes([0.2, 0.025, 0.1, 0.04])
    Fieldbutton = Button(Fieldax, 'Director', hovercolor='0.975')
    Circleax = fig.add_axes([0.35, 0.025, 0.1, 0.04])
    Circlebutton = Button(Circleax, 'R - Detection', hovercolor='0.975')
    Saveax = fig.add_axes([0.65, 0.025, 0.1, 0.04])
    Savebutton = Button(Saveax, 'Save plot', hovercolor='0.975')
    
    # upadate functions for buttons
    def reset(event):
        w_slider.reset()
        R_slider.reset()
        Thresh_slider.reset()
    button.on_clicked(reset)
    

    def finish(event):
        global defect_char
        # The displayed image is just one frame, now the whole stack is computed.
        print('Computing the whole stack...')
        e_vec, err_vec, cost_vec, theta_vec, phi, defect_char = can.get_anisotropy(imgpath, False, R_slider.val/bin_, round(1.5*w_slider.val), round(w_slider.val/bin_factor), fsig, BoxSize, Thresh_slider.val, peak_threshold, plotit=False, stack=stack, savedir = None)
        plt.close(fig)
        print('Done. You can save the data:')
        #print(defect_char['x'])

        fold = filedialog.asksaveasfile(defaultextension='.csv') # the user choses a place in file explorer
        defect_char.to_csv(fold.name) # the DataFrame is saved as csv
        print('Saved')

            
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
        axsave.set_title('feature size = %.0f px, R = %.0f px, order threshold = %.2f'%(w_slider.val, R_slider.val, Thresh_slider.val))
        
        fold = filedialog.asksaveasfile(defaultextension='.png') # make the user choose a file location
        figsave.savefig(fold.name) # save figure at this location
        plt.close(figsave)
        print('Saved')
    Savebutton.on_clicked(ClickSave)
    
    ### Where the infinite loop allow us to not be garbage collected (if I understand well) ###
    # it stops when the display window is closed: either by user or when OK is clicked on
    plt.show()
    while plt.fignum_exists(fig.number):
        plt.pause(0.1)
        
    return defect_char, w_slider.val, R_slider.val, Thresh_slider.val
        