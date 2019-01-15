#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 13:57:22 2018

@author: pch1g13

2+1-dimensional plotting functions.
"""

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as col
import numpy as np
from decimal import Decimal

def DecimalExponent(number):
    """
    Base-10 exponent of number.
    """
    (sign, digits, exponent) = Decimal(number).as_tuple()
    return len(digits) + exponent - 1

def DecimalMantissa(number):
    """
    Base-10 mantissa of number.
    """
    return Decimal(number).scaleb(-DecimalExponent(number)).normalize()

def PlotScalar2D(data,
                 xs,
                 ys,
                 simulation_directory,
                 output_directory,
                 filename,
                 fig_size=[8,4.5],
                 fig_dpi=480,
                 imshow_kwargs={},
                 labels={},
                 verbose=False):
    """
    Make plot of 2D scalar field and save to disk.

    Args:
    data - array(float) - scalar data to plot
    xs - array(float) - x coordinates of domain
    ys - array(float) - y coordinates of domain
    simulation_directory - str - absolute location of simulation data
    output_directory - str - relative location to save figure
    filename - str - filename to save with
    
    Kwargs:
    fig_size=[8,4.5] - list(float) - size in inches of figure
    fig_dpi=480 - int - resolution of figure to save
    imshow_kwargs={} - dict - optional arguements passed to imshow
    labels={} - dict - labels for plot, see below for expected values
    verbose=False - bool - print messages to IO

    Possible values in labels:
    xAxis - label for X-Axis
    yAxis - label for Y-Axis
    Title - title for figure
    Time - time label for figure
    Colourbar - label for colourbar, if not present, no colourbar will be added
    """
    ### Set up Matplotlib ###
    plt.ioff()
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family'] = 'STIXGeneral'
    
    ### Determine extent of data ###
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    imshow_extent = np.array([np.min(xs) - 0.5*dx,
                              np.max(xs) + 0.5*dx,
                              np.min(ys) - 0.5*dy,
                              np.max(ys) + 0.5*dy])
    
    imshow_size = np.array([imshow_extent[1]-imshow_extent[0],
                            imshow_extent[3]-imshow_extent[2]])
    
    imshow_kwargs["extent"] = imshow_extent
    
    data_max = np.max(data)
    if "vmax" in imshow_kwargs:
        if verbose:
            if data_max>imshow_kwargs["vmax"]:
                print("Density outside specified limits, clipping will occur.")
                print("Maximum data value is " + str(data_max) + " , imshow vmax is " + str(imshow_kwargs["vmax"]))
    else:
        imshow_kwargs["vmax"] = data_max
        
    data_min = np.min(data)
    if "vmin" in imshow_kwargs:
        if verbose:
            if data_min<imshow_kwargs["vmin"]:
                print("Density outside specified limits, clipping will occur.")
                print("Minimum data_value is " + str(data_min) + " , imshow vmin is " + str(imshow_kwargs["vmin"]))  
    else:
        imshow_kwargs["vmin"] = data_min
    
    ### Create Figure ###
    fig = plt.figure()
    fig.set_size_inches(fig_size[0],fig_size[1])
    ax = fig.add_subplot(1,1,1)
    
    im = ax.imshow(data.T,**imshow_kwargs)
    clb = fig.colorbar(im)
    
    if "Colourbar" in labels:
        clb.set_label(labels["Colourbar"])
    tick_locs = np.linspace(imshow_kwargs["vmax"],imshow_kwargs["vmin"],num=9)
    tick_labs = np.zeros(len(tick_locs),dtype=object)
    for i in range(len(tick_locs)):
        tick_labs[i] = r"$%.3f$" % tick_locs[i]
    clb.set_ticks(tick_locs)
    clb.set_ticklabels(tick_labs)
    
    if "xAxis" in labels:
        ax.set_xlabel(labels["xAxis"])
    if "yAxis" in labels:
        ax.set_ylabel(labels["yAxis"])
    if "Title" in labels:
        ax.set_title(labels["Title"])
    if "Time" in labels:
        ax.set_title(labels["Time"],
                     fontdict={'fontsize': 'medium',
                               'fontweight' : 'normal',
                               'verticalalignment': 'baseline',
                               'horizontalalignment': 'left'},loc='left')
    fig.savefig(simulation_directory + output_directory + filename + ".png",dpi=fig_dpi, bbox_inches = "tight")

    plt.clf()
    plt.close("all")
    
    return None

def PlotScalarLog2D(data,
                    xs,
                    ys,
                    simulation_directory,
                    output_directory,
                    filename,
                    fig_size=[8,4.5],
                    fig_dpi=480,
                    imshow_kwargs={},
                    labels={},
                    verbose=False):
    """
    Make plot of log10 of 2D scalar field and save to disk.

    Args:
    data - array(float) - scalar data to plot
    xs - array(float) - x coordinates of domain
    ys - array(float) - y coordinates of domain
    simulation_directory - str - absolute location of simulation data
    output_directory - str - relative location to save figure
    filename - str - filename to save with
    
    Kwargs:
    fig_size=[8,4.5] - list(float) - size in inches of figure
    fig_dpi=480 - int - resolution of figure to save
    imshow_kwargs={} - dict - optional arguements passed to imshow
    labels={} - dict - labels for plot, see below for expected values
    verbose=False - bool - print messages to IO

    Possible values in labels:
    xAxis - label for X-Axis
    yAxis - label for Y-Axis
    Title - title for figure
    Time - time label for figure
    Colourbar - label for colourbar, if not present, no colourbar will be added
    """
    
    ### Set up Matplotlib ###
    plt.ioff()
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family'] = 'STIXGeneral'
    
    ### Determine extent of data ###
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    imshow_extent = np.array([np.min(xs) - 0.5*dx,
                              np.max(xs) + 0.5*dx,
                              np.min(ys) - 0.5*dy,
                              np.max(ys) + 0.5*dy])
    
    imshow_size = np.array([imshow_extent[1]-imshow_extent[0],
                            imshow_extent[3]-imshow_extent[2]])
    
    imshow_kwargs["extent"] = imshow_extent
    
    data_max = np.log10(np.max(data))
    if "vmax" in imshow_kwargs:
        if verbose:
            if data_max>imshow_kwargs["vmax"]:
                print("Density outside specified limits, clipping will occur.")
                print("Maximum log(data) is " + str(data_max) + " , imshow vmax is " + str(imshow_kwargs["vmax"]))
    else:
        imshow_kwargs["vmax"] = data_max
        
    data_min = np.log10(np.min(data))
    if "vmin" in imshow_kwargs:
        if verbose:
            if data_min<imshow_kwargs["vmin"]:
                print("Density outside specified limits, clipping will occur.")
                print("Minimum log(data) is " + str(data_min) + " , imshow vmin is " + str(imshow_kwargs["vmin"]))  
    else:
        imshow_kwargs["vmin"] = data_min
    
    ### Create Figure ###
    fig = plt.figure()
    fig.set_size_inches(fig_size[0],fig_size[1])
    ax = fig.add_subplot(1,1,1)
    
    im = ax.imshow(np.log10(data).T,**imshow_kwargs)
    clb = fig.colorbar(im)
    
    if "Colourbar" in labels:
        clb.set_label(labels["Colourbar"])
    tick_locs = np.linspace(imshow_kwargs["vmax"],imshow_kwargs["vmin"],num=9)
    tick_labs = np.zeros(len(tick_locs),dtype=object)
    for i in range(len(tick_locs)):
        tick_labs[i] = r"$%.2f \times 10^{" % DecimalMantissa(10**Decimal(tick_locs[i])) + str(DecimalExponent(10**Decimal(tick_locs[i]))) + r"}$"
    clb.set_ticks(tick_locs)
    clb.set_ticklabels(tick_labs)
    
    if "xAxis" in labels:
        ax.set_xlabel(labels["xAxis"])
    if "yAxis" in labels:
        ax.set_ylabel(labels["yAxis"])
    if "Title" in labels:
        ax.set_title(labels["Title"])
    if "Time" in labels:
        ax.set_title(labels["Time"],
                     fontdict={'fontsize': 'medium',
                               'fontweight' : 'normal',
                               'verticalalignment': 'baseline',
                               'horizontalalignment': 'left'},loc='left')
    
    fig.savefig(simulation_directory + output_directory + filename + ".png",dpi=fig_dpi, bbox_inches = "tight")

    plt.clf()
    plt.close("all")
    
    return None

def PlotVelRhoRGB2D(speed,
                    direction,
                    log_density,
                    xs,
                    ys,
                    simulation_directory,
                    output_directory,
                    filename,
                    data_kwargs={},
                    fig_size=[8,4.5],
                    fig_dpi=480,
                    imshow_kwargs={},
                    labels={},
                    verbose=False):
    """
    Make plot of velocity and log_density, and save to disk.

    Args:
    speed - array(float) - speed data to plot
    direction - array(float) - direction data to plot
    log_density - array(float) - log10 of density data
    xs - array(float) - x coordinates of domain
    ys - array(float) - y coordinates of domain
    simulation_directory - str - absolute location of simulation data
    output_directory - str - relative location tscalaro save figure
    filename - str - filename to save with
    
    Kwargs:
    fig_size=[8,4.5] - list(float) - size in inches of figure
    fig_dpi=480 - int - resolution of figure to save
    imshow_kwargs={} - dict - optional arguements passed to imshow
    labels={} - dict - labels for plot, see below for expected values
    verbose=False - bool - print messages to IO

    Possible values in labels:
    xAxis - label for X-Axis
    yAxis - label for Y-Axis
    Title - title for figure
    Time - time label for figure
    Colourbar - label for density colourbar
    """
    
    ### Set up Matplotlib ###
    plt.ioff()
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family'] = 'STIXGeneral'
    
    ### Determine extent of data ###
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    imshow_extent = np.array([np.min(xs) - 0.5*dx,
                              np.max(xs) + 0.5*dx,
                              np.min(ys) - 0.5*dy,
                              np.max(ys) + 0.5*dy])
    
    imshow_size = np.array([imshow_extent[1]-imshow_extent[0],
                            imshow_extent[3]-imshow_extent[2]])
    
    imshow_kwargs["extent"] = imshow_extent
    
    speed_max = np.max(speed)
    if "speed_max" in data_kwargs:
        if verbose:
            if speed_max>data_kwargs["speed_max"]:
                print("Speed outside specified limits, clipping will occur.")
                print("Maximum speed value is " + str(speed_max) + " , speed_max is " + str(data_kwargs["speed_max"]))
        speed_max = data_kwargs["speed_max"]
        
    speed_min = np.min(speed)
    if "speed_min" in data_kwargs:
        if verbose:
            if speed_min<data_kwargs["speed_min"]:
                print("Speed outside specified limits, clipping will occur.")
                print("Minimum speed value is " + str(speed_min) + " , speed_min is " + str(data_kwargs["speed_min"]))
        speed_min = data_kwargs["speed_min"]
    
    log_rho_max = np.max(log_density)
    if "log_rho_max" in data_kwargs:
        if verbose:
            if log_rho_max>data_kwargs["log_rho_max"]:
                print("Density outside specified limits, clipping will occur.")
                print("Maximum log_density value is " + str(log_rho_max) + " , log_rho_max is " + str(data_kwargs["log_rho_max"]))
        log_rho_max = data_kwargs["log_rho_max"]
        
    log_rho_min = np.min(log_density)
    if "log_rho_min" in data_kwargs:
        if verbose:
            if log_rho_min<data_kwargs["log_rho_min"]:
                print("Density outside specified limits, clipping will occur.")
                print("Minimum log_density value is " + str(log_rho_min) + " , log_rho_min is " + str(data_kwargs["log_rho_min"]))
        log_rho_min = data_kwargs["log_rho_min"]
    
    ### Create Figure ###
    fig = plt.figure()
    fig.set_size_inches(fig_size[0],fig_size[1])
    
    hsv_array = np.zeros((len(xs),len(ys),3))
    hsv_array[:,:,0] = np.clip((direction + 2.0*np.pi*(direction<0.0))/(2*np.pi),0.0,1.0) # Hue - Direction mapped from 0 to 2pi
    hsv_array[:,:,1] = np.clip((speed-speed_min)/(speed_max-speed_min),0.0,1.0) # Saturation - Speed mapped from speed_min to speed_max c
    hsv_array[:,:,2] = np.clip((log_density - log_rho_min)/(log_rho_max - log_rho_min),0.0,1.0) # Value - log(rho) mapped from 1e-10.5 to 1e-3
    rgb_array = col.hsv_to_rgb(hsv_array)
    
    ax1 = fig.add_axes([0.05,0.05,0.5,0.9])
    ax2 = fig.add_axes([0.7,0.45,0.3,0.4],projection='polar')
    ax3 = fig.add_axes([0.65,0.05,0.4,0.3])
    
    im = ax1.imshow(np.swapaxes(rgb_array,0,1),**imshow_kwargs)

    theta = np.linspace(0.0,2.0*np.pi,num=257)
    r = np.linspace(0.0,1.0,num=50)
    T,R = np.meshgrid(theta,r)
    ax2.contourf(T,R,T,256,cmap="hsv")
    ax2.set_xticklabels([r"$0$",r"$\pi/4$",r"$\pi/2$",r"$3\pi/4$",r"$\pi$",r"$-3\pi/4$",r"$-\pi/2$",r"$-\pi/4$"])
    ax2.set_yticks(np.array([]))
    ax2.text(np.pi/2,1.35,r"Direction",
             horizontalalignment="center",
             verticalalignment="baseline",
             fontsize="large")
    
    ### Saturation/Value plot
    sats = np.linspace(0.0,1.0,num=256)
    vals = np.linspace(0.0,1.0,num=256)
    Sats, Vals = np.meshgrid(sats,vals)
    hsv_array_hs = np.zeros((Sats.shape[0],Sats.shape[1],3))
    hsv_array_hs[:,:,0] = np.zeros(Sats.shape)
    hsv_array_hs[:,:,1] = Sats
    hsv_array_hs[:,:,2] = Vals
    rgb_array_hs = col.hsv_to_rgb(hsv_array_hs)
    ax3.imshow(rgb_array_hs,origin="lower",interpolation="bessel",aspect=0.75)
    
    ax3.set_xticks([0,64,128,192,256])
    tick_locs_S = np.linspace(speed_min,speed_max,num=5)
    tick_labs_S = []
    for i in range(len(tick_locs_S)):
        tick_labs_S.append("$" + r"{:.3f}".format(tick_locs_S[i]) + "$")
    ax3.set_xticklabels(tick_labs_S)
    ax3.set_xlabel(r"$|v| \left[ c \right]$")
    
    ax3.set_yticks([0,64,128,192,256])
    tick_locs_V = np.linspace(log_rho_min,log_rho_max,num=5)
    tick_labs_V = np.zeros(len(tick_locs_V),dtype=object)
    for i in range(len(tick_locs_V)):
        tick_labs_V[i] = r"$%.2f \times 10^{" % DecimalMantissa(10**Decimal(tick_locs_V[i])) + str(DecimalExponent(10**Decimal(tick_locs_V[i]))) + r"}$"
    ax3.set_yticklabels(tick_labs_V)
    
    if "Colourbar" in labels:
        ax3.set_ylabel(labels["Colourbar"])
    
    if "xAxis" in labels:
        ax1.set_xlabel(labels["xAxis"])
        
    if "yAxis" in labels:
        ax1.set_ylabel(labels["yAxis"])
        
    if "Title" in labels:
        ax1.set_title(labels["Title"])
        
    if "Time" in labels:
        ax1.set_title(labels["Time"],
                      fontdict={'fontsize': 'medium',
                                'fontweight' : 'normal',
                                'verticalalignment': 'baseline',
                                'horizontalalignment': 'left'},loc='left')
    
    fig.savefig(simulation_directory + output_directory + filename + ".png",dpi=fig_dpi, bbox_inches = "tight")

    plt.clf()
    plt.close("all")
    
    return None
