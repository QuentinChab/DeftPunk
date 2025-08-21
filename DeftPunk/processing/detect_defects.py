# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:17:45 2024

@author: Quentin (from Carles Matlab code charac_defectes)
"""

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.signal import convolve2d
#import anisotropy_functions as fan
from skimage import measure
import matplotlib.pyplot as plt
import warnings

def defect_detection(theta, coherency, fov, BoxSize, order_threshold, peak_threshold, plotall=False, method='weighted'):


    if method == 'weight':
        prop_name = 'centroid_weighted'
    else:
        prop_name = 'centroid'
        
        
    boxes=[]
    chargeb=[]
    defect_axis=[]
    
    # get the field of x-orientation and y-orientation
    mvec_x1=np.cos(theta)
    mvec_y1=np.sin(theta)
    Angle_Director = np.arctan(mvec_y1/mvec_x1) #just theta ? But projected on one part of the trigo circle
    
    # Qloc evaluate whether orientation field is well-defined at each point with a window of size fov            
    Qloc=np.sqrt(gaussian_filter(np.cos(2*theta),fov)**2+gaussian_filter(np.sin(2*theta),fov)**2)
    #ndir = np.arctan2(uniform_filter(np.sin(2*theta), fov), uniform_filter(np.cos(2*theta), fov))/2
    #thdif = np.arctan2(np.sin(2*ndir-2*theta), np.cos(2*ndir-2*theta))/2
    #s = (3*np.cos(thdif)**2-1)/2
    #Qloc=gaussian_filter(s, fov)
    if plotall:
        sh = Qloc.shape
        X, Y = np.meshgrid(range(sh[1]), range(sh[0]))
        plt.figure()
        ax = plt.subplot(111, projection='3d')
        ax.plot_surface(X, Y, Qloc)
    
    # binariN uses a threshold to segregate well-defined and ill-defined points
    # the threshold is a mutliple of the minimum Qloc
    absolute_threshold = order_threshold#min(order_threshold*np.min(Qloc), 0.5)
    binariN=measure.label(Qloc<absolute_threshold)

    # use the binary filter to get the different ill-defined regions
    coherency[coherency==0] = 0.001
    weight = 1/np.array(coherency)
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")  # catch all warnings
        
        props = measure.regionprops_table(binariN, intensity_image=weight , properties=('centroid', 'bbox', 'centroid_weighted', 'intensity_min', 'coords'))
        
        if w and any(issubclass(warning.category, RuntimeWarning) for warning in w):
            print("No defect is detected on this image")

    #

    
    BigBoxSize  = int(20)
    centroidsN  = np.ones((len(props['bbox-0']),2))
    chargeb     = np.ones((len(props['bbox-0'])))
    defect_axis = np.ones((len(props['bbox-0'])))
    boxes       = np.ones((len(props['bbox-0']),4), dtype=int)
    boxesp      = np.ones((len(props['bbox-0']),4), dtype=int)
    sh          = Angle_Director.shape
    if plotall: plt.figure()
    
    for s in range(len(props['bbox-0'])):
        centroidsN[s, :] = [props[prop_name+'-0'][s], props[prop_name+'-1'][s]] 
        boxesp[s,:] = [max(int(centroidsN[s,0]-BoxSize/2),0),min(int(centroidsN[s,0]+BoxSize/2),sh[0]), max(int(centroidsN[s,1]-BoxSize/2),0),min(int(centroidsN[s,1]+BoxSize/2),sh[1])] #little box size, used right now
        if method=='min':
            boxQloc = Qloc[boxesp[s,0]:boxesp[s,1],boxesp[s,2]:boxesp[s,3]]
            centroidsN[s,:] = np.array(np.unravel_index(boxQloc.argmin(), boxQloc.shape)) + np.array((boxesp[s,0], boxesp[s,2]))
            boxesp[s,:] = [max(int(centroidsN[s,0]-BoxSize/2),0),min(int(centroidsN[s,0]+BoxSize/2),sh[0]), max(int(centroidsN[s,1]-BoxSize/2),0),min(int(centroidsN[s,1]+BoxSize/2),sh[1])] #little box size, used right now
            
        
        boxes[s,:] = [int(centroidsN[s,0]-BigBoxSize/2),int(centroidsN[s,0]+BigBoxSize/2), int(centroidsN[s,1]-BigBoxSize/2),int(centroidsN[s,1]+BigBoxSize/2)] #little box size, used right now
        angles_temp=Angle_Director[boxesp[s,0]:boxesp[s,1],boxesp[s,2]:boxesp[s,3]] #only the selected region
        cycle = np.array([*angles_temp[0,:], *angles_temp[1:,-1], *np.flip(angles_temp[-1,:-2]), *np.flip(angles_temp[1:-2,0])]) % np.pi # borders of the region
        
        absolute_threshold = peak_threshold#­min(0.9, max(0.5, peak_threshold*np.max(np.diff(cycle))))
        chargeb[s]=(np.sum(np.diff(cycle)>np.pi*absolute_threshold)*(-np.pi)+np.sum(np.diff(cycle)<(-np.pi*peak_threshold))*(np.pi)+np.sum(np.diff(cycle)))/(2*np.pi) #sum(diff(cycle)) is usually 0, account for the cases where psi(theta=0)=0 
        if plotall: plt.plot(np.diff(cycle)/np.pi, label='Detection')
        #if plotall: plt.plot(np.arange(len(cycle)), cycle, '.', label='Defect %.0f'%(s+1))
    if plotall:
        plt.plot(plt.xlim(), [0,0], 'k')
        plt.plot(plt.xlim(), [absolute_threshold,absolute_threshold], '--k', label='peak_threshold')
        plt.plot(plt.xlim(), [-absolute_threshold,-absolute_threshold], '--k')
        plt.xlabel('Box contour')
        plt.ylabel(r'$\Delta\theta_{Box}/\pi$ [rad]')
        #plt.ylabel(r'$\theta_{Box}$ [rad]')
        plt.legend()
        plt.tight_layout()
        
    if plotall:
        plt.figure()
        ax = plt.imshow(Qloc, cmap='Reds')
        plt.quiver(np.cos(theta), np.sin(theta), angles='xy', headaxislength=0, headlength=0, pivot='mid', width=0.1, units='xy', scale=1.2)
        plt.plot(centroidsN[:, 1], centroidsN[:,0], 'o')
        plt.colorbar(ax, label='Order Parameter')
        plt.figure()
        ax = plt.imshow(binariN)
        plt.quiver(np.cos(theta), np.sin(theta), angles='xy', headaxislength=0, headlength=0, pivot='mid', width=0.1, units='xy', scale=1.2)
        plt.colorbar(ax)
    #compute orientation of the defect axis
    if True: #previously used method
        Qxxm=(mvec_x1*mvec_x1-mvec_y1*mvec_y1)/2.
        Qxym=mvec_y1*mvec_x1
        DyQxx,DxQxx = np.gradient(Qxxm)
        DyQxy,DxQxy = np.gradient(Qxym)
        
        for s in range(len(chargeb)):
            if np.abs(chargeb[s]-1/2)<0.2:
                chargeb[s] = 0.5
                V1=(DxQxy-DyQxx)[boxesp[s,0]:boxesp[s,1],boxesp[s,2]:boxesp[s,3]]
                V2=(DxQxx+DyQxy)[boxesp[s,0]:boxesp[s,1],boxesp[s,2]:boxesp[s,3]]
                defect_axis[s]=np.arctan2(np.mean(V1),np.mean(V2))
            elif np.abs(chargeb[s]+1/2)<0.2:
                chargeb[s] = -0.5
                V1=(-DxQxy-DyQxx)[boxesp[s,0]:boxesp[s,1],boxesp[s,2]:boxesp[s,3]]
                V2=(DxQxx-DyQxy)[boxesp[s,0]:boxesp[s,1],boxesp[s,2]:boxesp[s,3]]
                defect_axis[s]=-np.arctan2(np.mean(V1),np.mean(V2))/3
            else:
                defect_axis[s]=np.nan
    # else: #doostmohammadi method
        # for s in range(len(chargeb)):
            
        #     detR = np.min([BoxSize, centroidsN[s,1], centroidsN[s,0], theta.shape[1]-centroidsN[s,1]-1, theta.shape[0]-centroidsN[s,0]-1])
        #     if detR<4:
        #         defect_axis[s]=np.nan
        #     else:
        #         # iX, iY = findParametrizedCirclePoint(detR, [posx[n], posy[n]])
        #         # nemlist = np.empty(len(iX))
        #         # for i in range(len(nemlist)):
        #         #     nemlist[i] = theta[round(iY[i]),round(iX[i])]
        #         # XposRel = centroidsN[0] - iX
        #         # YposRel = centroidsN[1] - iY
        #         # philist = np.arctan2(YposRel,XposRel)
                
        #         philist, nemlist = fan.compute_angle_diagram(theta, detR, [centroidsN[s,1], centroidsN[s,0]])
                
        #         FT = np.sum(np.exp(complex(0,1)*philist)*np.cos(2*nemlist))
        #         defect_axis[s]=np.nan = -np.angle(FT)
        #         if np.abs(chargeb[s]-1/2)<0.2:
        #             defect_axis[s] = -np.angle(FT)
        #         elif np.abs(chargeb[s]+1/2)<0.2:
        #             defect_axis[s] = np.angle(FT)/3

                
                
    return Qloc, boxes, chargeb, defect_axis, centroidsN 
    


def findParametrizedCirclePoint(circRad, cpos):
    
    density = np.pi/4 #Starting with some multiple of pi ensures symmetry of the resulting path
    trialParam = np.arange(0, 2*np.pi, density)
    trialX = np.round(circRad * np.cos(trialParam))
    trialY = np.round(circRad * np.sin(trialParam))
    trialDists = np.sqrt((np.square(np.diff(trialX))) + (np.square(np.diff(trialY))))
    

    
    while np.any(trialDists > np.sqrt(2)):
        density = density/2
        trialParam = np.arange(0, 2*np.pi, density)
        trialX = np.round(circRad * np.cos(trialParam))
        trialY = np.round(circRad * np.sin(trialParam))
        trialDists = np.sqrt((np.square(np.diff(trialX))) + (np.square(np.diff(trialY))))
    
    #Remove repeated points in final path
    # print(trialX)
    trialPoints, rightind = np.unique([trialX,trialY],axis=1,return_index=True)
    # print(trialPoints)
    trialPoints = trialPoints[:,np.argsort(rightind)]
    # print(trialPoints)
    # print(trialX)
    # print(trialY)
    
    #Add specified displacement (based on center of circle).
    outX = np.round(trialPoints[0,:] + cpos[0])
    outY = np.round(trialPoints[1,:] + cpos[1])

    return outX, outY

# def rebin(arr, new_shape):
#     arr_shape = arr.shape
#     remx = arr_shape[0]%20
#     remy = arr_shape[1]%20
#     if remx>0:
#         lastcol = arr[-1,:]
#         repeated = np.repeat(lastcol[:,np.newaxis],20-remx,1)
#         arr = np.concatenate((arr, repeated), axis=0)
#     if remy>0:
#         lastlin = arr[:,-1]
#         repeated = (np.ones((20-remy,1))*lastlin).transpose()
#         arr = np.concatenate((arr, repeated), axis=1)
    
#     shape = (new_shape[0], arr.shape[0] // new_shape[0],
#               new_shape[1], arr.shape[1] // new_shape[1])
#     return arr.reshape(shape).mean(-1).mean(1)


# #for test purpose
# BoxSize = int(8)
# peak_threshold = 0.2
# order_threshold = 0.2
# fov =  2
# orientationTable = np.loadtxt(r'C:\Users\Quentin\Documents\Analysis\Defect_detection_test\orientation_tables\Exercise_5\Fig3_s10_grid20.csv', delimiter = ',', skiprows=1)
# image = plt.imread(r'C:\Users\Quentin\Documents\Analysis\Defect_detection_test\images\Exercise_5\Fig3.tif')
# theta_list = orientationTable[:,5]
# x = orientationTable[:,0]
# y = orientationTable[:,1]
# Ny = np.where(np.diff(x)<0)[0][0]+1
# theta = np.reshape(theta_list, (-1, Ny))
# image = rebin(image, theta.shape)