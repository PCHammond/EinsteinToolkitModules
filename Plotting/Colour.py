#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 16:38:39 2018

@author: pete

Functions for colourspaces.
"""

import numpy as np
from colour import CCT_to_uv, STANDARD_OBSERVERS_CMFS
import warnings

def uvToxyY(uv):
    """
    Convert from uv colourspace to xyY.
    """
    x = 3*uv[0]/(2*uv[0] - 8*uv[1] + 4)
    y = 2*uv[1]/(2*uv[0] - 8*uv[1] + 4)
    return np.array([x,y,1.0])

def GetCIExyYfromuv(T,D_uv = 0.00322335):
    """
    Convert from uv(T) to CIExyY.
    """
    cmfs = STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer']
    if T>500.0:
        if T<2e5:
            uv = CCT_to_uv(T, D_uv=D_uv, cmfs=cmfs)
        else:
            uv = CCT_to_uv(2e5, D_uv=D_uv, cmfs=cmfs)
    else:
        uv = CCT_to_uv(500.0, D_uv=D_uv, cmfs=cmfs)
    return uvToxyY(uv)
    
def GetCIExyY(T):
    """
    Get CIExyY(T).
    """
    T3_arr = np.array([(10**9)/(T**3),(10**6)/(T**2),(10**3)/(T),1.0])
    if (1667.0<=T)and(T<=4000.0):
        x = np.sum(np.array([-0.2661239,-0.2343589,0.8776956,0.179910]) * T3_arr)
    elif (4000.0<=T)and(T<=25000.0):
        x = np.sum(np.array([-3.0258469,2.1070379,0.2226347,0.240390]) * T3_arr)
    else:
        raise ValueError
    
    x3_arr = np.array([x**3,x**2,x,1.0])
    if (1667.0<=T)and(T<=2222.0):
        y = np.sum(np.array([-1.1063814,-1.34811020,2.18555832,-0.20219683]) * x3_arr)
    elif (2222.0<=T)and(T<=4000.0):
        y = np.sum(np.array([-0.9549476,-1.37418593,2.09137015,-0.16748867]) * x3_arr)
    elif (4000.0<=T)and(T<=25000.0):
        y = np.sum(np.array([3.0817580,-5.87338670,3.75112997,-0.37001483]) * x3_arr)
    else:
        raise ValueError
    
    return np.array([x, y, 1.0])

def GetsRGBfromxyY(xyY):
    """
    Convert from xyY to SRGB.
    """
    X = xyY[2] * xyY[0] / xyY[1]
    Y = xyY[2]
    Z = xyY[2] * (1-xyY[0]-xyY[1]) / xyY[1]
    
    M = np.array([[ 3.2406,-1.5372,-0.4986],
                  [-0.9689, 1.8758, 0.0415],
                  [ 0.0557,-0.2040, 1.0570]])
    RGBl = np.matmul(M,np.array([X,Y,Z]))
    
    sRGB = np.zeros(3)
    for i in range(3):
        if RGBl[i]<=0.0031308:
            sRGB[i] = 12.92*RGBl[i]
        else:
            sRGB[i] = (1.0-0.055)*RGBl[i]**(1.0/2.4) - 0.055
    
    return sRGB

def GetsRGBfromTemp(T):
    """
    Wrapper for GetCIExyYfromuv and GetsRGBfromxyY.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message='"domain" and "range" variables have different size, "range" variable will be resized to "domain" variable shape!')
        xyY = GetCIExyYfromuv(T)
    sRGB = GetsRGBfromxyY(xyY)
    return np.clip(sRGB, 0.0, 1.0)

def Linmap(vals):
    """
    Remap vals from 0 at min(vals) to 1.0 at max(vals).
    """
    return (vals - vals.min())/(vals.max() - vals.min())

def test(n=1001):
    Ts = np.logspace(2,9,num = n)
    cols = np.zeros((n,3))
    for i in range(n):
        cols[i] = GetsRGBfromTemp(Ts[i])
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(np.array([cols]),aspect="auto")
    plt.figure()
    plt.plot(np.sum(cols,axis=-1))
    return None