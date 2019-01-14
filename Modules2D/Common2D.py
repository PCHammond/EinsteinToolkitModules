#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 16:46:53 2018

@author: pch1g13
"""
import sys, os.path
ETM_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(ETM_dir)

from Common import GetKeysForRefinement_Level
import numpy as np

def GetDomainAfterTransforms(xs,ys,transforms):
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
    dataset_size = np.asarray(dataset.shape)[::-1]
    delta = dataset.attrs['delta']
    buffer = dataset.attrs['cctk_nghostzones']
    bl = dataset.attrs['origin'] + buffer*delta
    tr = bl + (dataset_size-(2*buffer))*delta
    return np.array([bl[0],tr[0],bl[1],tr[1]]) ##[xmin,xmax,ymin,ymax]

def GetDatasetExtentIncGhost2D(dataset):
    dataset_size = np.asarray(dataset.shape)[::-1]
    delta = dataset.attrs['delta']
    bl = dataset.attrs['origin']
    tr = bl + (dataset_size - np.array([1,1]))*delta
    return np.array([bl[0],tr[0],bl[1],tr[1]]) ##[xmin,xmax,ymin,ymax]

def GetSmallestCoveringRL2D(list_rLs,list_keys,xs,ys,file_input):
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