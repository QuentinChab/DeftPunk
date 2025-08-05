# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 18:28:44 2024

@author: Quentin

This script contains only one functions: defect_analyzer.
It's the highest function of the hierarchy: it only treats interface
"""
import matplotlib.pyplot as plt
import numpy as np
from ..processing.CallAll import one_defect_anisotropy
from ..processing.compute_anisotropy import crop_rotate_scalar
import DeftPunk.processing.OrientationPy as OPy
import trackpy as tp
from scipy import stats
from scipy.optimize import curve_fit
import os

origin_file = os.path.abspath( os.path.dirname( __file__ ) )


bin_factor = 4




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
        for i in range(len(field)):
            tpset = pset[pset['frame']==i]
            tmset = mset[mset['frame']==i]
            for ip in range(len(tpset)):
                xcrop, ycrop, rot_field = crop_rotate_scalar(field[i], axis=-tpset['axis'].iloc[ip], cropsize=cropsize, xcenter=tpset['x'].iloc[ip], ycenter=tpset['y'].iloc[ip])
                patterns_p[pincrement] = rot_field
                pincrement +=1
                # plt.figure()
                # plt.imshow(rot_field, cmap='binary')
            for im in range(len(tmset)):
                xcrop, ycrop, rot_field = crop_rotate_scalar(field[i], axis=-tmset['axis'].iloc[im], cropsize=cropsize, xcenter=tmset['x'].iloc[im], ycenter=tmset['y'].iloc[im])
                patterns_m[mincrement] = rot_field
                mincrement +=1
    else:

        for ip in range(len(pset)):
            xcrop, ycrop, rot_field = crop_rotate_scalar(field, axis=-pset['axis'].iloc[ip], cropsize=cropsize, xcenter=pset['x'].iloc[ip], ycenter=pset['y'].iloc[ip])
            patterns_p[ip] = rot_field
        for im in range(len(mset)):
            xcrop, ycrop, rot_field = crop_rotate_scalar(field, axis=-mset['axis'].iloc[im], cropsize=cropsize, xcenter=mset['x'].iloc[im], ycenter=mset['y'].iloc[im])
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
        e, err_e, costmin, theta_unit = one_defect_anisotropy(orientation, R=R, xc=table['x'].iloc[i]/2, yc=table['y'].iloc[i]/2, axis = table['axis'].iloc[i], plotit=ref)
        ref = False
        th_list.append(theta_unit)
    
    theta = np.arctan2(np.nanmean(np.sin(th_list), axis=0), np.nanmean(np.cos(th_list), axis=0))
    return e, theta

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
    dangle_flat = []
    SD_flat = []
    
    fset = []
    
    #polar plot histogram and trajectory schematic
    
    # f_traj = plt.figure()
    # plt.quiver(-1,0, label='Head-to-tail direction')
    
    for i in range(len(part_list)):
        datapart = datap[datap['particle']==part_list[i]]
        vx = np.diff(datapart['x'], dt)*unit_per_px/unit_per_frame
        vy = np.diff(datapart['y'], dt)*unit_per_px/unit_per_frame
        axis = datapart['axis'].to_numpy()[:-dt]
        dangle = np.arccos((vx*np.cos(axis) + vy*np.sin(axis))/np.sqrt(vx**2+vy**2))
    #     dangle_list[i] = dangle
        dangle_flat = [*dangle_flat, *dangle]
        vamp = np.sqrt(vx**2 + vy**2)
        
        SD_flat = [*SD_flat, *vamp]
    #     #build traj
    #     xtraj = np.zeros(len(dangle)+1)
    #     ytraj = np.zeros(len(dangle)+1)
    #     for j in range(1,len(xtraj)):
    #         xtraj[j] = (xtraj[j-1]+vamp[j-1]*np.cos(dangle[j-1]))
    #         ytraj[j] = (ytraj[j-1]+vamp[j-1]*np.sin(dangle[j-1]))
    #     plt.plot(xtraj, ytraj)
        
    # plt.xlabel('x ['+unit_space+']')
    # plt.ylabel('y ['+unit_space+']')
    # plt.legend()
    # plt.title('Defect trajectory with respect to defect axis')
    # plt.tight_layout()
    # fset.append(f_traj)
    
    f_polar, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    circular_hist(ax, np.array(dangle_flat), bins=16, density=True, offset=0, gaps=True)
    plt.title('Angle between defect axis and velocity \n over %.0f points.\n Frequency proportionnal to box area.'%(dt))
    plt.tight_layout()
    fset.append(f_polar)
    
    # plot diffusion 
    
    Npart = [ np.sum(part_vec==part) for part in part_list]
    dt_list = np.arange(1, np.max(Npart))
    MSD = np.empty((len(part_list), len(dt_list)))
    MSD_std = np.empty((len(part_list), len(dt_list)))
    for it in range(len(dt_list)):
        SD_list = []
        for ip in range(len(part_list)):
            datapart = datap[datap['particle']==part_list[ip]]
            x = datapart['x'].to_numpy()
            y = datapart['y'].to_numpy()
            fr = datapart['frame'].to_numpy()
            sdt = []
            for j in range(len(fr)):
                sel = fr-fr[j]==dt_list[it]
                if np.any(sel):
                    ind = np.where(sel)[0][0]
                    sdt.append((x[ind]-x[j])**2+(y[ind]-y[j])**2)
            vx = np.diff(datapart['x'], dt_list[it])*unit_per_px/unit_per_frame
            vy = np.diff(datapart['y'], dt_list[it])*unit_per_px/unit_per_frame
            SD = np.square(vx) + np.square(vy)
            SD_list = [*SD_list, *SD]
            MSD[ip,it] = np.nanmean(sdt)*unit_per_frame*unit_per_frame
            MSD_std[ip,it] = np.nanstd(sdt)*unit_per_frame*unit_per_frame
    
    def linear_model(log_t, alpha, log_A):
        return log_A + log_t*alpha
    
    dt_list = dt_list*unit_per_frame
    err_RMSD = MSD_std/2/MSD
    err_logRMSD = err_RMSD/np.sqrt(MSD)
    err_logRMSD[err_logRMSD==0] = np.nan
    alphas = np.empty(len(part_list))*np.nan
    bs = np.empty(len(part_list))*np.nan
    ralphas = np.empty(len(part_list))*np.nan
    rbs = np.empty(len(part_list))*np.nan
    for j in range(len(part_list)):
        # print(np.any(np.isnan(np.log(dt_list[3:-3]))))
        # print(np.all(np.isnan(np.log(MSD[3:-3])/2)))
        sel2 = np.logical_not(np.isnan(MSD[j,3:-3]))
        params, cov = curve_fit(linear_model, np.log(dt_list[3:-3][sel2]), np.log(MSD[j,3:-3][sel2])/2, bounds=(0,np.inf), maxfev=int(1e5))#, p0=(0.5, np.nanmean(np.log(MSD)/np.log(dt_list))))#, sigma=err_logRMSD, absolute_sigma=True, ))
        cerr = np.sqrt(np.diag(cov))
        alphas[j] = params[0]
        if params[1]==0:
            params[1] = np.nan
        bs[j] = np.exp(params[1])
        ralphas[j] = cerr[0]
        rbs[j] = cerr[1]*bs[j]
    # alpha, log_A = params
    # if log_A == 0:
    #     log_A = np.nan
    # alpha_err, log_A_err = np.sqrt(np.diag(cov))
    # A = np.exp(log_A)
    # A_err = A * log_A_err
    # D = A**2 / 4
    # D_err = 2*A*A_err/4
    
    D = bs**2 / 4
    #D_err = 2*bs*rbs/4
    
    # params, cov = curve_fit(linear_model, dt_list, np.sqrt(MSD), maxfev=int(1e4), p0=(0.5, np.nanmean(np.sqrt(MSD/dt_list))))#, sigma=err_RMSD, absolute_sigma=True)
    # alpha, A= params
    # alpha_err, A_err = np.sqrt(np.diag(cov))
    # D = A**2/4
    # D_err = 2*A*A_err/4
    
    # fitted_RMSD = A * dt_list**alpha
    # fitted_RMSD = bs * dt_list**alphas
    
    f_dif = plt.figure()
    for ip in range(len(part_list)):
        plt.plot(dt_list, MSD[ip,:], '.')
        plt.plot(dt_list, bs[ip]*dt_list**alphas[ip], '--')    
    plt.plot([], [], 'k.', label='Data')
    plt.plot([], [], 'k--', label='Fit')
    # plt.plot(dt_list, np.sqrt(MSD), '+', label='Data')
    # plt.fill_between(dt_list, np.sqrt(MSD)-np.sqrt(MSD_std), np.sqrt(MSD)+np.sqrt(MSD_std), alpha=0.5, color=plt.gca().lines[-1].get_color())
    # plt.plot(dt_list, fitted_RMSD, label='Fit: RMSD = %.3f$\\cdot\\tau^{%.3f}$'%(A, alpha))
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Time delay $\\tau$ ['+unit_t+']')
    plt.ylabel('MS Displacement ['+unit_space+']')
    plt.legend()
    #plt.title('The diffusion coefficient is $D=%.3f\\pm %.3f$\n The diffusion exponent is $\\alpha=%.3f\\pm %.3f$'%(D, D_err, alpha, alpha_err))
    plt.tight_layout()
    fset.append(f_dif)
    
    f_D = plt.figure()
    plt.hist(D)
    plt.xlabel('Diffusion coefficient ['+unit_space+'$^2$/'+unit_t+']')
    plt.ylabel('Counts')
    plt.tight_layout()
    fset.append(f_D)
    
    f_hist = plt.figure()
    plt.hist(SD_flat, bins=20)
    plt.xlabel('Velocity amplitude ['+unit_space+'/'+unit_t+']')
    plt.ylabel('Counts')
    plt.tight_layout()
    fset.append(f_hist)
    
    return fset

def temporal_analysis(traj, min_length = 20, N_defects=3):
    traj = tp.filter_stubs(traj, 20)
    part = traj['particle']
    npart = np.unique(part)
    partcharge = np.empty(npart.shape)
    isolation = np.empty(npart.shape)
    for i in range(len(npart)):
        charge = traj['charge'][part==npart[i]]
        partcharge[i] = np.round(2*np.nanmean(charge))/2
        isolation[i] = np.nanmean(traj['MinDist'][part==npart[i]])
    
    ppart = npart[partcharge==1/2]
    isolation_rank = ppart[np.argsort(isolation[partcharge==1/2])]
    
    plt.figure()
    for i in range(N_defects):
        traji = traj[traj['particle']==isolation_rank[i]]
        plt.errorbar(traji['frame']*15, traji['Anisotropy'], traji['Error'], fmt='o', label='Defect %.0f'%(i))
    plt.xlabel('Time [min]')
    plt.ylabel('Anisotropy []')
    plt.legend()
    plt.tight_layout()
    
    for i in range(N_defects):
        traji = traj[traj['particle']==isolation_rank[i]]
        plt.figure()
        ax = plt.gca()
        #ax2 = ax.twinx()
        ax.errorbar(traji['frame']*15, traji['Anisotropy'], traji['Error'], fmt='o', label='Anisotropy of defect %.2f'%(i))
        #ax2.plot(traji['frame']*15, traji['MinDist'], 'r+-', label='Distance to neighbor')
        ax.set_ylabel('Anisotropy []')
        #ax2.set_ylabel('Distance [px]')
        plt.title('Defect #%.0f'%(i))
        plt.xlabel('Time [min]')
        #ax.legend()
        #ax2.legend()
        plt.tight_layout()
   