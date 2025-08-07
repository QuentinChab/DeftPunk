//////////
// 08/10/2024
// Quentin Chaboche
// PhD student at Institut Curie UMR168
// quentin.chaboche@curie.fr
//
// Last Update: 05/08/2025
//////////

DeftPunk is a Python package that provides tools and graphical interfaces for analyzing images of nematic systems. It identifies and tracks topological defects (such as ±1/2, ±1 charges), estimates their orientation, and measures the anisotropy between splay and bend elastic moduli.
These tools are particularly useful for researchers working in soft matter physics, liquid crystals, or active matter, aiming to quantitatively characterize defect dynamics in experimental or simulated systems.

In this README, you will find:
1. How to install the package (requirements)
2. How to run the main interfaces/functions
3. Description of outputs
4. Description of parameters and interface sliders/buttons
5. Detailed description of selected functions
6. Known issues
7. What are the files in the DeftPunk folder?

//////////////////// 1. How to become a Defect Punk ///////////////////

The necessary packages are:
- Standard Python libraries (matplotlib, numpy, pandas and scipy)
- Trackpy
- scikit-image

Use pip to install the required packages:
> pip install trackpy
> pip install -U scikit-image

Versions I use (05/08/2025):
python 3.13.5
numpy 2.3.1
matplotlib 3.10.0
pandas 2.3.1
scipy 1.16.0
trackpy 0.7
scikit-image 0.25.2


//////////////////// 2. Run main interfaces and functions ///////////////////
You can use DeftPunk in three different ways, depending on your preferences and platform.
Notes
- The interface (method 1) does not work on Mac.
- Make sure to add the DeftPunk folder to your Python path or run from the correct working directory.

GUI Interface method (see main.py):
[Windows/Linux only]
Interactive analysis with minimal coding.
Open the main interface using:
> from DeftPunk import detect_defect_GUI
> _ = detect_defect_GUI()
This will launch an interface for loading images, choosing parameters visually.
See section 4 for detailed buttons and sliders.

Semi-programmatic (see Example.py):
> from DeftPunk import defect_analyzer
> defect_char, det_param, vfield, _ = defect_analyzer(
>    imgpath="path/to/image.tif",
>    det_param=[feature_size, R, order_threshold],
>    stack=True,
>    frame=0
>    savedir='path/savepath'
> )
Opens an interface to tune parameter from input image.

Fully programmatic (See Example_No_GUI.py):
> from DeftPunk import analyze_image
> e_vec, err_vec, cost_vec, theta_vec, phi, defect_char = analyze_image(
>     imgpath="path/to/image.tif",
>     feature_size=10,
>     R=20,
>     order_threshold=0.7,
>     plotit=True,
>     savedir="path/to/output/"
> )


//////////////////// 3. Description of the outputs ///////////////////
After analysis, DeftPunk can save a .csv and a .txt file at chosen location.

Table (.csv) defect data from detection. The fields are:
- x and y:
    coordinates of the defect center (in pixels)
- charge:
    Topological charge of the defect: -1, -1/2, +1/2 or +1
- axis:
    for half-interger defects, angle between the x direction and the defect axis. For +1/2 it is the tail, and for -1/2 one of the branch.
- Anisotropy:
    only for +1/2 defects. It quantifies wether it has more of a V shape (Anisotropy=-1) or a U shape (Anisotropy=+1).
    it relates to different terms in the nematic energy of a system. 
    The energetic cost for director line to splay has modulus k1, the energetic cost for director line to bend has modulus k3.
    Then Anisotropy = (k1-k3)/(K1+k3)
    This is a material parameter probed on +1/2 defects.
- Error:
    estimated error on Anisotropy. 
- MinDist:
    in pixel, distance to nearest defect (whatever the charge)
- frame:
    for stacks, frame number
- particle:
    when tracking has been performed, tracking id.

Parameter file (.txt) such that:
The name is [chosen_name]_parameters.txt
At each save, marks date and time, as well as detection and/or tracking parameters chosen.
If the same name is used multiple time, new entries are appended.


//////////////////// 4. Description of parameters and interface sliders and buttons ///////////////////
Here is the description of the detection parameters:
- feature_size f:
    this is the typical size of an oriented element in the image (example: width of a cell)
    It controls:
    1- The computation of the director angle (window size = 1.5*f)
    2- Downsampling of the director field with respect to pixels (downsampling = f/4)
    this influences the detection of defects
    it is well chosen if the director field is well detected
- order_threshold o:
    between 0 and 1
    after computation of the director field, the nematic order parameter s is computed with averaging director angle over a region
    if a region has s < order_threshold, we consider that one defect is present
    if it is too low, you may miss some defects
    if it is too high, you may have false positives and have two defect "fuse" (the s < threshold region associated with each defect fuse together)
- R:
    the Anisotropy is computed with the shape of the defect
    the shape is represented as the director angle along a contour around the defect
    R is the radius of the circular contour
    it should be as large as possible to reduce uncertainty
    it should not detect director deformation due to features other than the defect
    
Here are the tracking parameters:
- search_range sR:
    in pixel, the maximum defect displacement between two frames
- memory:
    how many missing frames do we accept in a trajectory
    (we can link a particle in frame n with one in frame n+memory)
- filt:
    we filter out trajectories if they have less than filt frames
    
// Now the 3 interfaces are described:
Main Interface (called with `detect_defect_GUI`)
Buttons
- Load:
    Open a browser to load an image to analyze
    You can also load a dataset to then skip analysis
    You can as well load a director field (may be unstable)
- Start Detection:
    Open detection interface
- Check tracking:
    Open tracking interface
- Save Data:
    Save the dataset with current detection
- Apply on directory:
    Open a browser for user to select a folder. 
    The detection is applied to every image in the folder with previously chosen parameters
- Statistics:
    Apply some statistics to the computed data
Fields
- Let you chose the units and conversion factors for time and space
Slider
- frame:
    for a stack, chose the frame that will display for parameter selection
    
Detection interface (called with `defect_analyzer()`)
Buttons
- Invert Color:
    black-and-white or white-and-black
- Director:
    display director angle field on top of the image
- R-detection:
    display around each defect the contour on which Anisotropy is computed (see description of Anisotropy)
- Reset:
    to initial parameters
- Save Image:
    open a browser for user to save the displayed image
- OK:
    Parameters selection is complete. If it is a stack all frames are computed and data is returned.
Sliders:
- see previous section on parameters

Tracking Interface (called with `check_tracking()`)
Buttons
- OK:
    Parameters selection is complete. Return them.
- Save Movie:
    Save the stack with tracks on top on a tif file (selected by user)
- Save Dataset:
    open a browser to save data
- preview movie:
    with current parameters, display the movie and tracks
- loop:
    if ticked, the movie keeps looping when you press preview movie
Sliders:
- see tracking parameters previous section



/////////////// 5. Detailed description of selected functions //////////////////////
We will detail all inputs of 
1. defect_analyzer
2. analyze_image
3. trackmap
You can also have documentation with help([function_name])

//
1. defect_analyzer
This calls the detection interface (see previous section)
> defect_char, det_param, vfield, _ = defect_analyzer(imgpath, det_param, stack, frame, [optionnal_paramters])

INPUTS
- imgpath, string
    path to the image to analyze
- det_param, [float, float, float]
        list of initial detection parameters, in order [feature_size, R, order_threshold]
- stack, boolean
        is the image a stack? Default True
- frame, int
        if it is a stack, which frame to analyze? Default 0th
- um_per_px, float
        conversion between physical unit and px. Default 1
- unit, string
        physical unit. Default pixel.
- vfield, array or None
        user-provided director field. If None (default) it is computed.
- endsave, boolean
        Do we save the data at the end of execution? Default True.
- savedir, str
        if endsave==True, where is the data saved?
        if 'Select', opens a browser to select where to save it.
    
OUTPUTS
- defect_char, pandas DataFrame
        table of detected defects. Fields described in section 3.
- det_param, [float, float, float]
        list of selected detection parameters, [feature_size, R, order_threshold]
- vfield, array of floats
        computed director field
- list_of_buttons
        list containing the buttons, to keep them active
    
//
2. analyze_image
This function takes the path of an image, compute the orientation field, finds the defect and estimates the anisotropy of the +1/2 types
It requires the detection parameters. 

> e_vec, err_vec, cost_vec, theta_vec, phi, defect_char = analyze_image(imgpath, feature_size, R, order_threshold, prescribed_field=None, plotit=False, stack=False, savedir = None, give_field=False)

INPUTS
- imgpath : string
        Path to the image.
- feature_size : number
        as described in section 4
- R : number
        as described in section 4
- order_threshold : float, optional
        as described in section 4
- prescribed_field : array or None, optional
        If provided the used director field.
        If None (Default) the director is computed within the function.
- plotit : boolean, optional
        Do we display detetion steps? Default is False.
- stack : boolean, optional
        is the image a stack. Default is False.
- savedir : string or None, optional
        path where to save the data.
        if None (default), the data is not saved.
- give_field : boolean, optional
        if True, the director field computed is returned.
        Default is False.
        
OUTPUTS
- e_vec : 1D numpy array
        Array of anisotropies for all detected defects
- err_vec : 1D numpy array
        corresponding error.
- cost_vec : 1D numpy array
        the anisotropy is fitted on the shape. This is the cost function of the best fit.
- th_vec : list
        Corresponding angular profiles, defining the shape. This is the director field as a function of the contour around the defect.
- phi : list
        Azimuthal angle corresponding to th_vec. The same for all profiles.
- defect_char : pandas DataFrame
        DataFrame containing the result of detection. See section 3 
- orientation : 2D numpy array
        Array of angles of the director field. Only returned if give_field=True.
- pos : list of 2 1D arrays
        Corresponding coordinates of each angle from orientation ([x, y]). Only returned if give_field=True.

3. trackmap
From a image stack and a defect dataset (as generated by the code), create a tif stack with movie and tracks on it.

> trackmap(img, data, savedir=np.nan, filt=np.nan, yes_traj=True):

INPUTS
- img : array
        image stack
- data : pandas DataFrame
        data from detection, as returned by the different functions and described in section 3
- savedir : str or nan
        where to save the resulting stack
        if NaN (default), nothing happens
- filt : numeric, optional
        re-apply a tracking filter, as described in section 4
- yes_traj : boolean, optional
        if yes (Default), the tracks are displayed.
        if False, only the points are.

/////////////// 6. known issues ////////////////////////////////////////////////////
- The preferred image format is .tif. Format .png is tested for non-stack images. jpg and bmp should work too.
    Other format are not tested. It should work for non-stack images but not for stacks.

- On Mac, calling the browser makes the kernel crash.
    Avoid using load button and all save buttons
    do not use global interface

- On Windows, calling the browser may open blank windows. Does not impair functionality.

- On linux, calling the browser cause warning messages to appear. Ignore them.

////////////// 7. What's in the folder //////////////////////////////////////////

Here is a description of the files in the top folder.

// First two tif images to test the package on
`20240215_Actin_Only_8_tif.tif`
Microscopy image of actin on a substrate of lipids.
Taken by Gwendal Guérin, Institut Curie UMR168

`MT_kinesin_blue.tif`
Stack of a system of Microtubule+Kinesin+ATP.
From Dogic group: Sanchez, Chen, DeCamp, Heyman and Dogic, Nature 491 431-434 (2012)

// Then 3 example functions to show how to use the package
`main.py`
Most simple use of the package. Calls the global interface.

`Example.py`
Semi-programmatic use. 
The user programatically defines the image path and save path, and select parameter from interface.

`Example_No_GUI.py`
Programmatic use. Image path, save path and parameters are defined by the user in the script.

// Finally descriptive files
`README.md`
You're reading it. How-to-use instructions.

`ScriptDescription.txt`
More detailed description of certain functions.

// Directories
`/DeftPunk`
Contains all the logic, functions to call.

`Old`
Old version of the function.

`__pycache__`
Generated automatically when I open script in IDE.

////////////////// Hope you find it useful! /////////////////////////////////////////
