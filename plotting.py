import os
import pickle
from webbrowser import get
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as ss
# from ams_utilities import *
import npultra_utils as npu
import matplotlib.pyplot as plt
import matplotlib as mpl

sns.set_style('white')
params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
plt.rcParams.update(params)

def make_contour_plot(amps,ax,interp_scale=10,shape=(48,8),levels=10,vmin=0.0,vmax=1.0,cmap='viridis'):
    
    from scipy import interpolate
    
    FR = amps.reshape(shape[0],shape[1])[::]

    x= np.arange(shape[1])
    y = np.arange(shape[0])
    
    func = interpolate.interp2d(x,y,FR)
    
    xnew = np.linspace(0,shape[1],len(x)*interp_scale)
    ynew = np.linspace(0,shape[0],len(y)*interp_scale)
    X, Y = np.meshgrid(xnew, ynew)
    Z = func(xnew,ynew)

    # for spine in ['right','left','top','bottom']:
    #     ax.spines[spine].set_visible(False)

    CS = ax.contourf(X,Y,Z,
                     vmin= vmin,vmax=vmax,
                     levels = levels,
                     cmap=cmap,
                     extend='both')

    ax.set_yticklabels([])
    ax.set_xticklabels([])

def make_st_plot(unit,ax,shape=(48,8),vmin=-100,vmax=100,cbar=False):
    amps = get_amp(unit)
    max_chan = np.argmax(amps)
    min_idx = np.where(unit[max_chan,:]==np.min(unit[max_chan,:]))[0][0]

    if shape == (48,8):
        max_row = max_chan//8
        max_col = max_chan%8
        col_data = unit.reshape(shape[0],shape[1],unit.shape[1])[:,max_col,:][::-1].astype(float)
        
    if shape == (384,1):
        col_data = unit[::-1]

    if shape == (192,2):
        unit = unit.reshape(shape[0],shape[1],unit.shape[1]) #assume original data is channels x samples
        max_col = max_chan%shape[1]
        col_data = unit[:,max_col,:][::-1]

    sns.heatmap(col_data[:,min_idx-15:min_idx+45],vmin=vmin,vmax=vmax,cbar=cbar,cmap='seismic_r',cbar_kws={'label': '\u03BCV'},ax=ax)

def make_temporal_plot(unit,ax,max_chan,min_idx,top_chan='max',ylim=(-200,100)):

    wf = []

    for chan in unit:
        time = np.linspace(-0.5,1.5,60)
        ax.plot(time,chan[min_idx-15:min_idx+45],color='lightgray',alpha=0.2)

    if top_chan=='max':
        ax.plot(time,unit[max_chan,min_idx-15:min_idx+45],lw=1.5,color='red')
    elif top_chan=='avg':
        ax.plot(time,np.mean(wf,axis=0),color='k',lw=0.5,label='mean')
    elif top_chan=='both':
        ax.plot(time,unit[max_chan,min_idx-15:min_idx+45],lw=2,color='red',label='max amp')
        ax.plot(time,np.mean(wf,axis=0),color='k',lw=3,label='mean')

    ax.set_ylabel('\u03BCV',rotation=0,fontsize=12,labelpad=0,y=0.4)
    ax.set_xlabel('Time (ms)',fontsize=12)
    ax.set_ylim(ylim[0],ylim[1])
    ax.set_yticks([ylim[0],0,ylim[1]])
    for spine in ['left','bottom']:
        ax.spines[spine].set_linewidth(2)

def make_waveform_plot(unit,figsize=(4,6),fontsizes=(12,8),ylim=(-200,100)):

    fig = plt.figure(constrained_layout=False,figsize=figsize)
    gs = fig.add_gridspec(nrows=8, ncols=14)
    ax1 = fig.add_subplot(gs[0:5,0:3])
    ax2 = fig.add_subplot(gs[0:5,3:])
    ax3 = fig.add_subplot(gs[5:,3:])
    sns.despine(left=True,bottom=True,ax=ax1)
    sns.despine(ax=ax3)

    if unit.shape[0]>unit.shape[1]:
        pass
    else:
        unit = unit.T
    min_idx = np.where(unit==np.min(unit))[1][0]

    amps = get_amp(unit)
    max_chan = np.argmax(amps)
    make_contour_plot(amps,ax1,
                      interp_scale=10,
                      vmin=np.max(amps)*0.01,
                      vmax=np.max(amps),
                      levels = [np.max(amps)*0.1,
                                np.max(amps)*0.2,
                                np.max(amps)*0.3,
                                np.max(amps)*0.4,
                                np.max(amps)*0.5,
                                np.max(amps)*0.6,
                                np.max(amps)*0.7,
                                np.max(amps)*0.8,
                                np.max(amps)*0.9])
    make_st_plot(unit,ax2)
    make_temporal_plot(unit,ax3,max_chan,min_idx,ylim=ylim)
        
    for ax in [ax1,ax2]:
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        
    ax3.tick_params(axis='y',which='major',reset=True,right=False,labelsize=fontsizes[1])
    ax3.tick_params(axis='x',which='major',reset=True,top=False,labelsize=fontsizes[1])
    
    
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5)
    
    return fig,ax1,ax2,ax3

def interpolate_line(row,col,ang,Z):
    import scipy
    x,y = npu.find_intercept((row,col),(0,7),(0,47),ang)
    num =int(np.sqrt((x-row)**2 + (y-col)**2)*6)
    c,d = np.linspace(row, x, int(num)), np.linspace(col, y, int(num))

    zi = scipy.ndimage.map_coordinates(Z, np.vstack((d,c)))
    return(zi,c,d)

def make_directional_plot(unit,df):

    fig = plt.figure(figsize=(6,4))
    gs = fig.add_gridspec(nrows=10, ncols=10)
    ax1 = fig.add_subplot(gs[:,0:1])
    ax2 = fig.add_subplot(gs[:,4:])

    sns.despine(ax=ax2)

    amps = get_amp(unit).reshape(48,8)
    # amps = amps-np.mean(np.sort(amps.reshape(384,))[:10])
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2.xaxis.set_tick_params(top = False)
    ax2.yaxis.set_tick_params(right = False)

    ax2.set_xlabel('Distance from soma (\u03BCm)')
    ax2.set_ylabel('Max. amplitude\n(\u03BCV, interpolated)',y=0.5)
    ax2.set_ylim(0,(np.max(amps)*1000)+50)
        
    x = np.arange(8)
    y = np.arange(48)
    from scipy import interpolate
    func1 = interpolate.interp2d(x,y,amps)
    xnew = np.linspace(0,8,len(x))
    ynew = np.linspace(0,48,len(y))

    X, Y = np.meshgrid(xnew, ynew)
    Z = func1(xnew,ynew)

    CS = ax1.contourf(X,Y,Z,levels =np.linspace(0,0.2,11),vmin=0.01,cmap='magma',extend='both')

    max_chan = df['max_amp_channel']

    row,col = max_chan%8,max_chan//8

    colors = ['b','b','g','g']
    vals  = [df['dmin'],df['dmax'],df['tmin'],df['tmax']]
    labels = ['min(d)','max(d)','min(\u03C4)','max(\u03C4)']
    zs = []
    for i,ang in enumerate([df['theta_dmin'],df['theta_dmax'],df['theta_tmin'],df['theta_tmax']]):
        
        zi,c,d = interpolate_line(row,col,ang,amps)
        zs.append(len(zi))
        zd = np.linspace(0,len(zi),1000)
        interp_func2 = interpolate.interp1d(np.linspace(0,len(zi),len(zi)),zi) 
        zi_i = interp_func2(zd)
        try:
            ax2.scatter(vals[i],zi[round(vals[i])]*1000,s=40,color=colors[i],zorder=10)
        except:
            ax2.scatter(vals[i],zi[round(vals[i])-2]*1000,s=40,color=colors[i],zorder=10)

        if i%2==0:
            ax1.plot(c,d,color=colors[i])
            ax2.plot(np.linspace(0,len(zi),len(zi_i)),zi_i*1000,lw=2,color=colors[i],label = labels[i]) 
        else:      
            ax1.plot(c,d,color=colors[i],dashes=[3,3])
            ax2.plot(np.linspace(0,len(zi),len(zi_i)),zi_i*1000,lw=2,color=colors[i],label = labels[i],dashes=[3,3]) 
           
            
    ax2.axhline(30,color='r',dashes=[3,3],zorder=0)
    ax2.set_xlim(0,100)
    ax2.legend(frameon=False,fontsize=12)
    # ax2.set_yticklabels([0.0,50,100,150])
    # fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)

    return fig


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def get_PNW_cmap(palette='Starfish',n_colors = 10):

    '''Returns color map based on input string from a list. With credit to Jake Lawlor at https://github.com/jakelawlor/PNWColors/blob/master/R/PNWColors.R
    cmap: Starfish, Shuskan, Bay, Winter, Lake, Sunset, Shuskan2, Cascades, Sailboat, Moth, Spring, Mushroom, Sunset2, Anemone'''

    from matplotlib.colors import LinearSegmentedColormap as lsc
    import matplotlib as mpl
    Starfish = ['#24492e', '#015b58', '#2c6184', '#59629b', '#89689d', '#ba7999', '#e69b99']
    Shuskan = ['#33271e', '#74677e', '#ac8eab', '#d7b1c5', '#ebbdc8', '#f2cec7', '#f8e3d1', '#fefbe9']
    Bay = ['#00496f', '#0f85a0', '#edd746', '#ed8b00', '#dd4124']
    Winter = ['#2d2926', '#33454e', '#537380', '#81a9ad', '#ececec']
    Lake = ['#362904', '#54450f', '#45681e', '#4a9152', '#64a8a8', '#85b6ce', '#cde5f9', '#eef3ff']
    Sunset = ['#41476b', '#675478', '#9e6374', '#c67b6f', '#de9b71', '#efbc82', '#fbdfa2']
    Shuskan2 = ['#5d74a5', '#b0cbe7', '#fef7c7', '#eba07e', '#a8554e']
    Cascades = ["#2d4030","#516823","#dec000","#e2e260","#677e8e","#88a2b9"]
    Sailboat = ['#6e7cb9', '#7bbcd5', '#d0e2af', '#f5db99', '#e89c81', '#d2848d']
    Moth = ['#4a3a3b', '#984136', '#c26a7a', '#ecc0a1', '#f0f0e4']
    Spring = ['#d8aedd', '#bf9bdd', '#cb74ad', '#e69e9c', '#ffc3a3', '#fbe4c6']
    Mushroom = ['#4f412b', '#865a3c', '#ba783e', '#e69c4c', '#fbcc74', '#fffbda']
    Sunset2 = ['#1d457f', '#61599d', '#c36377', '#eb7f54', '#f2af4a']
    Anemone = ["#009474" ,"#11c2b5" ,"#72e1e1", "#f1f4ee" ,"#efddcf", "#dcbe9b" ,"#b0986c"]

    rgb_colors = [hex_to_rgb(c) for c in eval(palette)]
    rgb_colors = [(c[0]/255,c[1]/255,c[2]/255) for c in rgb_colors]
    cmap = lsc.from_list('cmap',rgb_colors)
    color_list = eval(palette)
    try:
        mpl.cm.register_cmap(palette, cmap)
    except:
        pass
    cp = sns.color_palette(palette,n_colors=n_colors)
    return color_list,cmap,cp

def make_color_palette(palette_name,palette,n_colors = 10):

    from matplotlib.colors import LinearSegmentedColormap as lsc
    import matplotlib as mpl

    rgb_colors = [hex_to_rgb(c) for c in palette]
    rgb_colors = [(c[0]/255,c[1]/255,c[2]/255) for c in rgb_colors]
    cmap = lsc.from_list('cmap',rgb_colors)
    color_list = palette
    try:
        mpl.cm.register_cmap(palette_name, cmap)
    except:
        pass
    cp = sns.color_palette(palette,n_colors=n_colors)
    return color_list,cmap,cp

def normalize(array):
    if np.max(array)>0:
        norm = (array - np.min(array)) / (np.max(array) - np.min(array))
    else:
        norm = array

    return norm

def get_trajectory(unit,pitch=6,shape=(48,8),threshold=0.05):
    amps = get_amp(unit)
    max_chan = np.argmax(amps)
    row,col = max_chan//shape[1],max_chan%shape[1]
    column = [chan for i,chan in enumerate(unit) if i%shape[1]==col]
    col_amps = get_amp(column)
    trajectory = []

    for ii,chan in enumerate(column):
        
        if ii>=np.argmax(col_amps):

            min_v = np.min(chan)
            min_idx = np.where(chan==min_v)[0][0]
            if min_v<=(np.min(column)*threshold):
                trajectory.append([ii,min_idx,min_v+ii*pitch])

    trajectory=np.array(trajectory)
    return trajectory

def make_ladder_plot(unit, probe_type='alpha', probe_shape=(48,8), figsize=(1.5,3), traj=True, thresh=0.1, plot=True,color='r',lw=[2,3]):
    
    # Initialize figure and axis if plotting is enabled
    if plot:
        fig, ax = plt.subplots(figsize=figsize)
        sns.despine(left=True, bottom=True)
        ax.set_xticks([])
        ax.set_yticks([])

    trajectory = []

    if probe_type == 'alpha':
        amps = get_amp(unit)
        max_chan = np.argmax(amps)
        row, col = max_chan // probe_shape[1], max_chan % probe_shape[1]
        column = [chan for i, chan in enumerate(unit) if i % probe_shape[1] == col]
        
        for ii, chan in enumerate(column):
            col_amps = get_amp(column)
            if plot:
                ax.plot(chan + ii * 6, color='k', lw=lw[0])
                if ii == np.argmax(col_amps):
                    ax.plot(chan + ii * 6, color=color, lw=lw[1], zorder=11)
            if ii >= np.argmax(col_amps):
                min_v = np.min(chan)
                min_idx = np.where(chan == min_v)[0][0]
                if min_v <= (np.min(column) * thresh):
                    if plot and traj:
                        ax.scatter(min_idx, min_v + ii * 6, color='green', s=20, zorder=10)
                    trajectory.append([ii, min_idx, min_v + ii * 6])
        
        trajectory = np.array(trajectory)
        if plot and traj:
            ax.plot(trajectory[:, 1], trajectory[:, 2], color='r', lw=lw[1])
        
    if probe_type == 'beta':
        amps = get_amp(unit[::])
        max_chan = np.argmax(amps)
        for ii, chan in enumerate(unit[::]):
            if plot:
                ax.plot(chan + ii * 3, color='k', lw=lw[0])
                if ii == max_chan:
                    ax.plot(chan + ii * 3, color='red', lw=lw[1], zorder=11)
            if ii >= max_chan:
                min_v = np.min(chan)
                min_idx = np.where(chan == min_v)[0][0]
                if min_v <= (np.min(unit[::]) * 0.01):
                    if plot and traj:
                        ax.scatter(min_idx, min_v + ii * 3, color='green', s=10, zorder=10)
                    trajectory.append([ii, min_idx, min_v + ii * 3])

        trajectory = np.array(trajectory)
        if plot and traj:
            ax.plot(trajectory[:, 1], trajectory[:, 2], color='r', lw=lw[1])

    # Only tighten layout and show plot if plotting is enabled
    if plot:
        fig.tight_layout()

    return trajectory,fig,ax

def plot_waveform_on_probe(unit, shape=(48,8),figsize=(4,24)):

    if unit.shape[0]>unit.shape[1]:
        pass
    else:
        unit = unit.T

    amps = get_amp(unit)
    peak_idx = [np.where(abs(chan)==np.max(abs(chan)))[0][0] for chan in unit]
    peaks = np.array([chan[peak_idx[i]] for i,chan in enumerate(unit)]).reshape(shape)
    min,max = np.min(unit),np.max(unit)

    fig,ax=plt.subplots(shape[0],shape[1],figsize=figsize)

    ax2 = fig.add_axes([0,0,1,1])
    sns.despine(ax=ax2,left=True,bottom=True)
    
    make_contour_plot(peaks,ax=ax2,interp_scale=1,shape=(48,8),levels=100,vmin=-500,vmax=500,cmap='seismic_r')

    # sns.heatmap(amps.reshape(48,8),ax=ax2,cbar=False,cmap='seismic_r')

    ax2.set_zorder(0)
    ax2.invert_yaxis()
    for i,axis in enumerate(ax.flatten()):
        sns.despine(ax=axis,left=True,bottom=True)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlim(0,len(unit[i]))
        axis.set_ylim(min,max)
        axis.plot(unit[i],lw=1,color='k')
        axis.patch.set_facecolor(None)
        axis.patch.set_alpha(0.0)
        axis.set_zorder(10)

    # ax2.set_facecolor("none")
    plt.margins(0.0)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.1,hspace=0.0,left=0.0,right=1.0,bottom=0.0,top=1.0)
    # plt.gca().set_position([0,0,1,1])
    
    return fig,ax

def get_CI(data,ci=0.95,axis=0):

    '''Take data in the form of a pandas dataframe or ndnumpy array and calculates the upper and lower bounds of a confidence interval along a declared axis'''

    from scipy.stats import norm
    z = norm.ppf((1 + ci) / 2)
    mean = np.nanmean(data,axis=axis)
    se = [z*((np.nanstd(data.loc[idx],ddof=1)/np.sqrt(len(data)))) for idx,row in enumerate(data.iterrows())]
    upperCI = [m + se[i] for i,m in enumerate(mean)]
    lowerCI = [m - se[i] for i,m in enumerate(mean)]

    return mean,upperCI,lowerCI