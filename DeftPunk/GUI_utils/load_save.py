#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 18:34:23 2025

@author: quentin

load_save.py

Helper functions to handle loading images and saving data.

There are 2 functions:
    - load_image
    - datasave

in DeftPunk package, GUI_utils subpackage

"""

import os
import tifffile as tf
import matplotlib.pyplot as plt
import numpy as np
import tkinter
import trackpy as tp
import datetime


def load_image(imgpath, channel=0):
    """
    From a path, load an image, determine if it is a stack and make into 
    a one-color image if necessary.
    
    > img, is_stack, units = load_image(imgpath)
    
    It is organized as such:
        - Handle tif file
        - Handle other files
        - If multichannel, reduce to one channel
    
    Parameters
    ----------
    imgpath : STR
        Path to the file.
    channel : int, optional
        If it is a multichannel image, make it into one channel.
        If 0 (default), average all channels. Otherwise select the channel_th channel

    Raises
    ------
    FileNotFoundError
        Fiele does not exist.
    NotImplementedError
        If the image format is not supported.

    Returns
    -------
    2D or 3D array
        Image or Image stack.
    Boolean
        True if the image is a stack, False otherwise.
    List
        Physical units encoded into metadata.
        In order [space_conversion, space_unit, time_conversion, time unit]
        If not in metadata: [1, 'px', 1, 'frame']

    """
    
    ## check if imgpath is correct
    if imgpath is None:
        return None, False, []
    
    if not os.path.exists(imgpath):
        raise FileNotFoundError(f"Image not found: {imgpath}")
        
    extension = imgpath.split('.')[-1] # extract image format
    
    # initialize units
    units = [1, 'px', 1, 'frame'] # default units
    
    stack = False
    
    ## first deal with tif format
    if extension in ['tif', 'tiff']:
        # Read TIFF metadata (to identify channels and stored units) using TiffFile
        with tf.TiffFile(str(imgpath)) as tif:
            axes = tif.series[0].axes  # E.g., "TZCYX"
            img = tif.asarray() # image 
            # If T it's a time stack, if Z it's a z-stack
            stack = ('T' in axes) or ('Z' in axes)
            
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
    elif extension in ['gif', 'npz', 'npy']: # we do not handle those ones
        raise NotImplementedError(f"{extension} format is not supported. Try tif, png, or a matplotlib-supported format.")            
    else: # png, jpg, bmp, are assumed not stack
        img = plt.imread(imgpath)
        
    # reduce dimension if it is a multichannel image
    if len(img.shape)>2+stack:
        if channel==0: # 0 codes for averaging channels
            img = np.mean(img, axis=-1)
        else: # otherwise the channel can be selected
            img = img[::,channel]
    
    return img, stack, units

def datasave(dchar, d_param, t_param=None, units=[1, 'px', 1, 'frame'], savedir=None):
    """
    Saves 2 files:
        1. The table containing detection information as .csv
        2. The detection/tracking parameters
    

    Parameters
    ----------
    dchar : Pandas DataFrame
        Table containing detection information: 'x', 'y', 'charge', 'axis', 'Anisotropy'
        And possibly 'frame' and 'particle'
    d_param : list  of 3 elements
        Detection parameters (in order feature_size, R, order_threshold).
        If None they are not recorded.
    t_param : list of 3 elements, optional
        Tracking parameter (in order search_range, memory, filt).
        If None (default) they are not recorded.
    units : Size 4 list, optional
        Space and time physical units. The default is [1, 'px', 1, 'frame'].
    savedir : STR, optional
        Path to save the data. If None (default) ask the user through file explorer.

    Returns
    -------
    None.

    """
    
    # if no parameter is provided
    if (d_param is None) and (t_param is None):
        print('No data provided to save')
    
    else: 
        # calls a file explorer that ask where to save the data
        if (savedir is None) or (savedir=='Select'):
            root = tkinter.Tk()
            root.withdraw()
            root.call('wm', 'attributes', '.', '-topmost', '1')  # Bring dialog to front (optional)
            fold = tkinter.filedialog.asksaveasfilename(defaultextension='.csv')
            root.destroy()
        else:
            fold = savedir
            
        
        if fold:
            unit_per_px, unit, unit_per_frame, unit_t = units
            
            
            #re-index particle column in order for mind satisfaction
            if 'particle' in dchar.columns:
                defect_char_to_save = tp.filter_stubs(dchar, t_param[2])
                part_vec = defect_char_to_save['particle'].to_numpy()
                part_list = np.unique(part_vec)
                for i in range(len(part_list)):
                    defect_char_to_save.loc[part_vec==part_list[i], 'particle']=i
            else:
                
                defect_char_to_save = dchar
            
            ## saves the table into a csv
            defect_char_to_save.to_csv(fold)
            
            ## save the parameters into a txt folder
            paramfile = '.'.join(fold.split('.')[:-1]) + '_parameters.txt'
            now_ = datetime.datetime.now()
            with open(paramfile, "a") as f:
                f.write('At '+str(now_)) # write the time
                # write detection parameters if they are provided
                if not d_param is None:
                    f.write('\nfeature size = %.0f '%(d_param[0]*unit_per_px)+unit)
                    f.write('\nnematic order threshold = %.2f '%(d_param[2]))
                    f.write('\nDetection Radius = %.0f '%(d_param[1]*unit_per_px)+unit)
                # write tracking parameter if they are provided
                if not (t_param is None):
                    f.write('\nsearch range = %.0f '%(t_param[0]*unit_per_px)+unit)
                    f.write('\nmemory = %.0f '%(t_param[1]*unit_per_frame) + unit_t)
                    f.write('\nfilter (minimum trajectory length) = %.0f '%(t_param[2]*unit_per_frame)+unit_t)
            print('Data Saved')
        else:
            print('Saving cancelled')
