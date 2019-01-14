#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 16:42:41 2018

@author: pch1g13
"""
import sys, os.path
ETM_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(ETM_dir)

from Common import GetKeysForRefinement_Level
from Modules2D.Common2D import GetDatasetExtent2D, GetDatasetExtentIncGhost2D

from scipy.interpolate import RectBivariateSpline
import numpy as np

def RotateAndCopyY_angle(data):
    shape_original = np.asarray(data.shape)
    shape_new = np.array([shape_original[0]*2 - 1, shape_original[1]])
    result = np.zeros(shape_new)
    np.copyto(result[shape_original[0]-1:,:],data)
    rotated_data = (data[::-1,::-1]+np.pi)
    mask = rotated_data>np.pi
    ranged_data = rotated_data - mask.astype(float)*2*np.pi
    np.copyto(result[:shape_original[0],:],ranged_data)
    return result

def RotateAndCopyY(data):
    shape_original = np.asarray(data.shape)
    shape_new = np.array([shape_original[0]*2 - 1, shape_original[1]])
    result = np.zeros(shape_new)
    np.copyto(result[shape_original[0]-1:,:],data)
    np.copyto(result[:shape_original[0],:],data[::-1,::-1])
    return result

def ReflectY(data):
    shape_original = np.asarray(data.shape)
    shape_new = np.array([shape_original[0]*2 - 1, shape_original[1]])
    result = np.zeros(shape_new)
    np.copyto(result[shape_original[0]-1:,:],data)
    np.copyto(result[:shape_original[0],:],data[::-1,:])
    return result

def Interp2DScalar(list_rLs,
                   list_keys,
                   xs,
                   ys,
                   file_input,
                   do_clip=False,
                   clip_bounds=np.array([0.0,1.0]),
                   k_spline=1,
                   transforms=None):
    
    output = np.zeros((len(xs),len(ys)))
    for rL in list_rLs:
        keys_rL = GetKeysForRefinement_Level(rL,list_keys)
        datasets_current = []
        for key in keys_rL:
            datasets_current.append(file_input[key])
        delta = datasets_current[0].attrs['delta'][0]
        for dataset in datasets_current:
            extent_current = GetDatasetExtent2D(dataset)
            extent_current_IG = GetDatasetExtentIncGhost2D(dataset)
            xs_domain = np.arange(extent_current_IG[0],extent_current_IG[1]+0.1*delta,delta)
            ys_domain = np.arange(extent_current_IG[2],extent_current_IG[3]+0.1*delta,delta)
            interp_spline = RectBivariateSpline(xs_domain,ys_domain,np.asarray(dataset).T,kx=k_spline,ky=k_spline)
            xs_mask = np.logical_and(extent_current[0]<=xs,xs<=extent_current[1])
            ys_mask = np.logical_and(extent_current[2]<=ys,ys<=extent_current[3])
            xs_current = np.extract(xs_mask,xs)
            ys_current = np.extract(ys_mask,ys)
            shift = np.array([np.argmax(xs_mask),np.argmax(ys_mask)])
            data_interp = interp_spline.__call__(xs_current,ys_current)
            np.copyto(output[shift[0]:shift[0]+len(xs_current),
                             shift[1]:shift[1]+len(ys_current)],
                             data_interp)
    if do_clip:
        np.clip(output,clip_bounds[0],clip_bounds[1],out=output)
    
    if transforms:
        for i in range(len(transforms)):
            if transforms[i] == "rotate_right_half":
                output = RotateAndCopyY(output)
            elif transforms[i] == "reflect_y":
                output = ReflectY(output)
            else:
                raise ValueError("Unknown tranformation requested")
            
    return output

def Interp2DSpeedDirection(list_rLs,
                           list_keys_velx,
                           list_keys_vely,
                           xs,
                           ys,
                           file_input_velx,
                           file_input_vely,
                           do_clip=[False,False],
                           clip_bounds=np.array([[0.0,1.0],[-np.pi,np.pi]]),
                           k_spline=1,
                           transforms=None):
    
    output_speed = np.zeros((len(xs),len(ys)))
    output_direc = np.zeros((len(xs),len(ys)))
    
    for rL in list_rLs:
        keys_rL_velx = GetKeysForRefinement_Level(rL,list_keys_velx)
        keys_rL_vely = GetKeysForRefinement_Level(rL,list_keys_vely)
        datasets_current_velx = []
        datasets_current_vely = []
        for key in keys_rL_velx:
            datasets_current_velx.append(file_input_velx[key])
        for key in keys_rL_vely:
            datasets_current_vely.append(file_input_vely[key])
        delta = datasets_current_velx[0].attrs['delta'][0]
        for ds_idx in range(len(datasets_current_velx)):
            dataset_velx = datasets_current_velx[ds_idx]
            dataset_vely = datasets_current_vely[ds_idx]
            extent_current = GetDatasetExtent2D(dataset_velx)
            extent_current_IG = GetDatasetExtentIncGhost2D(dataset_velx)
            xs_domain = np.arange(extent_current_IG[0],extent_current_IG[1]+0.1*delta,delta)
            ys_domain = np.arange(extent_current_IG[2],extent_current_IG[3]+0.1*delta,delta)
            interp_spline_velx = RectBivariateSpline(xs_domain,ys_domain,np.asarray(dataset_velx).T,kx=k_spline,ky=k_spline)
            interp_spline_vely = RectBivariateSpline(xs_domain,ys_domain,np.asarray(dataset_vely).T,kx=k_spline,ky=k_spline)
            xs_mask = np.logical_and(extent_current[0]<=xs,xs<=extent_current[1])
            ys_mask = np.logical_and(extent_current[2]<=ys,ys<=extent_current[3])
            xs_current = np.extract(xs_mask,xs)
            ys_current = np.extract(ys_mask,ys)
            shift = np.array([np.argmax(xs_mask),np.argmax(ys_mask)])
            data_interp_velx = interp_spline_velx.__call__(xs_current,ys_current)
            data_interp_vely = interp_spline_vely.__call__(xs_current,ys_current)
            data_interp_speed = np.sqrt(data_interp_velx**2 + data_interp_vely**2)
            data_interp_direc = np.arctan2(data_interp_vely,data_interp_velx)
            np.copyto(output_speed[shift[0]:shift[0]+len(xs_current),
                                   shift[1]:shift[1]+len(ys_current)],
                                   data_interp_speed)
            np.copyto(output_direc[shift[0]:shift[0]+len(xs_current),
                                   shift[1]:shift[1]+len(ys_current)],
                                   data_interp_direc)
            
    if do_clip[0]:
        np.clip(output_speed,clip_bounds[0,0],clip_bounds[0,1],out=output_speed)
    if do_clip[1]:
        np.clip(output_direc,clip_bounds[1,0],clip_bounds[1,1],out=output_direc)
        
    if transforms:
        for i in range(len(transforms)):
            if transforms[i] == "rotate_right_half":
                output_speed = RotateAndCopyY(output_speed)
                output_direc = RotateAndCopyY_angle(output_direc)
            #elif transforms[i] == "reflect_y":
            #    output_speed = ReflectY(output_speed)
            else:
                raise ValueError("Unknown tranformation requested")
    
    return output_speed, output_direc