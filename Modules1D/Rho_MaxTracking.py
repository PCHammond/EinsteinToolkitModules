#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 21:50:00 2018

@author: pete

Functions for tracking the location of the maxium density.
"""

import numpy as np

def GetMaxRhoLocs(simulation_directory,data_directory):
    """
    Get location of maximum density in full x,y,z where data is only available for 0<x rotated pi about Z-Axis.

    Args:
    simulation_directory - str - absolute location of simulation
    data_directory - str - relative location of merged .npy data for hydro_analysis-hydro_analysis_rho_max_loc.npy

    Returns:
    array(float)[times,3] - x,y,z coords of location of maximum density
    """
    data = np.load(simulation_directory + data_directory + "hydro_analysis-hydro_analysis_rho_max_loc.npy")

    times = data[:,1]
    xs_orig = data[:,2]
    ys_orig = data[:,3]
    zs = data[:,4]

    xs_fixed = np.zeros(len(xs_orig))
    ys_fixed = np.zeros(len(ys_orig))
    np.copyto(xs_fixed,xs_orig)
    np.copyto(ys_fixed,ys_orig)

    change_locs = []

    for i in range(1,len(xs_orig)-2):
        if ((xs_orig[i-1]>xs_orig[i])&(xs_orig[i]<xs_orig[i+1])):
            if xs_orig[i-1]>xs_orig[i+1]:
                change_locs.append(i)
            else:
                change_locs.append(i-1)
        if ((xs_orig[i-1]>xs_orig[i])&(xs_orig[i]==xs_orig[i+1])&(xs_orig[i+1]<xs_orig[i+2])):
            change_locs.append(i)

    change_locs = np.asarray(change_locs)

    for loc in change_locs[::-1]:
        xs_fixed[loc+1:] *= -1.0
        ys_fixed[loc+1:] *= -1.0

    coords = np.zeros((len(times),3))
    coords[:,0] = xs_fixed
    coords[:,1] = ys_fixed
    coords[:,2] = zs
    
    return times, coords

def GetRadiiFromCoords(coords):
    """Get radii from coords [[x0,y0,z0],[x1,y1,z1],...,[xn,yn,zn]]."""
    return np.linalg.norm(coords,axis=1)

def GetFreqInstFromCoords(times, coords):
    """Get instantaneous frequency of rotation about Z from coords [[x0,y0,z0],[x1,y1,z1],...,[xn,yn,zn]]."""
    return np.gradient(np.unwrap(np.arctan2(coords[:,1],coords[:,0])))/np.gradient(times)/(2.0*np.pi)    