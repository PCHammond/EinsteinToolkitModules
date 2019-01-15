#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 16:46:53 2018

@author: pch1g13

Common function for 2+1 dimensional analysis.
"""
import sys, os.path
ETM_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(ETM_dir)

from Common import GetKeysForRefinement_Level
import numpy as np

def GetDomainAfterTransforms(xs,ys,transforms):
    """
    Determine size of simulation domain after transformations.
    
    Args:
    xs - array(float) - starting x coordinates
    ys - array(float) - starting y coordinates
    transforms - list(str) - list of transformation names in the order they are applied

    Returns:
    array(float) - transformed x coordinates
    array(float) - transformed y coordinates
    """
    xmin = np.min(xs)
    xmax = np.max(xs)
    ymin = np.min(ys)
    ymax = np.max(ys)
    
    x_samples = len(xs)
    y_samples = len(ys)
    
    for i in range(len(transforms)):
        if transforms[i] == "rotate_right_half":
            xmin = -xmax
            x_samples = 2*x_samples - 1
        elif transforms[i] == "reflect_y":
            xmin = -xmax
            x_samples = 2*x_samples - 1
        else:
            raise ValueError("Unknown tranformation requested")
    
    xs_output = np.linspace(xmin,xmax,num=x_samples)
    ys_output = np.linspace(ymin,ymax,num=y_samples)
    
    return xs_output, ys_output

def GetDatasetExtent2D(dataset):
    """
    Read extent of a dataset from HDF5 attributes.

    Args:
    dataset - HDF5 dataset - dataset for which extent is to be determined

    Returns:
    array(float) - extent of dataset in form [x_min, x_max, y_min, y_max]
    """
    dataset_size = np.asarray(dataset.shape)[::-1]
    delta = dataset.attrs['delta']
    buffer = dataset.attrs['cctk_nghostzones']
    bl = dataset.attrs['origin'] + buffer*delta
    tr = bl + (dataset_size-(2*buffer))*delta
    return np.array([bl[0],tr[0],bl[1],tr[1]]) ##[xmin,xmax,ymin,ymax]

def GetDatasetExtentIncGhost2D(dataset):
    """
    Read extent of a dataset including 'ghost zones' from HDF5 attributes.

    Args:
    dataset - HDF5 dataset - dataset for which extent is to be determined

    Returns:
    array(float) - extent of dataset in form [x_min, x_max, y_min, y_max]
    """
    dataset_size = np.asarray(dataset.shape)[::-1]
    delta = dataset.attrs['delta']
    bl = dataset.attrs['origin']
    tr = bl + (dataset_size - np.array([1,1]))*delta
    return np.array([bl[0],tr[0],bl[1],tr[1]]) ##[xmin,xmax,ymin,ymax]

def GetSmallestCoveringRL2D(list_rLs,list_keys,xs,ys,file_input):
    """
    Determine the smallest refinement level that entirely covers the domain.

    Args:
    list_rLs - list of refinement levels under consideration
    list_keys - list of HDF5 dataset keys
    xs - x coordinates for domain
    ys - y coordiantes for domain
    file_input - HDF5 file containg keys

    Returns:
    int - smallest refiinement level that covers domain
    """
    for rL in list_rLs[::-1]:
        keys_rL = GetKeysForRefinement_Level(rL,list_keys)
        Xs, Ys = np.meshgrid(xs,ys,indexing="ij")
        coverage = np.zeros((len(xs),len(ys)))
        for key in keys_rL:
            extent_dataset = GetDatasetExtent2D(file_input[key])
            coverage += np.logical_and(np.logical_and(extent_dataset[0]<=Xs,Xs<=extent_dataset[1]),np.logical_and(extent_dataset[2]<=Ys,Ys<=extent_dataset[3]))
        if np.all(coverage):
            return rL
    raise ValueError("Coarsest grid doesn't cover desired output.")