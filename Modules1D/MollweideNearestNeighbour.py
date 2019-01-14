#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 17:27:36 2018

@author: pch1g13

Functions for generating a nearest neighbour interpolation of an approximately equal-density spherical sample grid onto a latitude-longitude grid.
"""
### Add Python folder to sys.path ###
import sys
sys.path.append("/mainfs/home/pch1g13/Python")

import numpy as np
from EinsteinToolkitModules.Modules1D.GeodesicLibrary import GetGeodesicSamplePoints
from math import sqrt, ceil

### Coordinate transformations from theta,phi to lat,long
def ThetaToLat(thetas):
    return 0.5*np.pi-thetas

def PhiToLong(phis):
    return phis - 2.0*np.pi*(phis>np.pi)

def ThetaPhiToLatLong(coordsSpherical):
    coordsLatLong = np.zeros(coordsSpherical.shape)
    coordsLatLong[:,0] = ThetaToLat(coordsSpherical[:,0])
    coordsLatLong[:,1] = PhiToLong(coordsSpherical[:,1])
    return coordsLatLong

### Transformations from spherical angles to unit vectors
def ThetaPhisToUnitVec(thetaPhis):
    vecs = np.zeros((len(thetaPhis),3))
    vecs[:,0] = np.sin(thetaPhis[:,0])*np.cos(thetaPhis[:,1])
    vecs[:,1] = np.sin(thetaPhis[:,0])*np.sin(thetaPhis[:,1])
    vecs[:,2] = np.cos(thetaPhis[:,0])
    return vecs

def LatLonsToUnitVec(latLons):
    vecs = np.zeros((len(latLons),3))
    vecs[:,0] = np.cos(latLons[:,0])*np.cos(latLons[:,1])
    vecs[:,1] = np.cos(latLons[:,0])*np.sin(latLons[:,1])
    vecs[:,2] = np.sin(latLons[:,0])
    return vecs

### Find closest neighbours_vec to point_vec
def FindNearestNeighbour(point_vec,neighbours_vec):
    best_dot = -1.0
    best_neighbour = 0
    for i in range(len(neighbours_vec)):
        current_dot = np.dot(point_vec,neighbours_vec[i])
        if current_dot>best_dot:
            best_neighbour = i
            best_dot = current_dot
    return best_neighbour

def GetNearestNeighbourArrays(sampling_points,theta_samples,phi_samples,roundup=False):
    """
    Get nearest neighbour interpolation array from geodesic samples onto lat-long grid

    Args:
    sampling_points - int - requested number of samples for geodesic grid
    theta_samples - int - number of polar samples for lat-long grid
    phi_samples - int - number of azimuth samples for lat-long grid

    Kwargs:
    roundup=False - bool - if sampling points is not exact, round up or down

    Returns:
    array(float)[sampling_points*,2] - theta,phi coords of geodesic sample points
    array(float)[theta_samples,phi_samples] - index of nearest geodesic sample for each lat-long sample point
    array(float)[theta_samples+1,phi_samples+1] - Longitude of edges of lat-long grid with faces centred on lat-long samples
    array(float)[theta_samples+1,phi_samples+1] - Latitude of edges of lat-long grid with faces centred on lat-long samples
    """
    theta_min = 0.0
    theta_max = np.pi
    phi_min = -np.pi
    phi_max = np.pi
    
    if roundup:
        division = int(ceil(sqrt(sampling_points/20)))
    else:
        division = int(round(sqrt(sampling_points/20)))
    
    neighbour_samples = 20*division**2
    
    theta_edges = np.linspace(theta_min,theta_max,num=2*theta_samples+1)[::2]
    phi_edges = np.linspace(phi_min,phi_max,num=2*phi_samples+1)[::2]
    
    Theta_edges, Phi_edges = np.meshgrid(theta_edges,phi_edges,indexing="ij")
    
    geodesicSamplesCoordsThetaPhi, geodesicSamplesAreas = GetGeodesicSamplePoints(neighbour_samples)
    geodesicSamplesVec = ThetaPhisToUnitVec(geodesicSamplesCoordsThetaPhi)

    ThetaPhis = np.transpose(np.array([Theta_edges.flatten(),Phi_edges.flatten()]),(1,0))
    
    SampleVecs = ThetaPhisToUnitVec(ThetaPhis)
    NearestNeighbours = np.zeros(len(SampleVecs),dtype=int)
    
    for i in range(len(SampleVecs)):
        NearestNeighbours[i] = FindNearestNeighbour(SampleVecs[i],geodesicSamplesVec)
        
    NearestNeighbours = np.reshape(NearestNeighbours,(theta_samples+1,phi_samples+1))
    
    lat_edges = ThetaToLat(theta_edges)
    lon_edges = PhiToLong(phi_edges)
    Lon_edges, Lat_edges = np.meshgrid(lon_edges,lat_edges)
    
    return geodesicSamplesCoordsThetaPhi, NearestNeighbours, Lon_edges, Lat_edges