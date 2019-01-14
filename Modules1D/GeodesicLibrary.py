#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 23:06:38 2018

@author: pete

Functions for creation of geodesic grids in 3-dimensions.
"""

import numpy as np
from math import sqrt, floor, ceil

### Rotation Matricies
def Rot_X(theta):
    """Rotation matrix for rotation about X-Axis."""
    return np.array([[1.0,          0.0,           0.0],
                     [0.0,np.cos(theta),-np.sin(theta)],
                     [0.0,np.sin(theta), np.cos(theta)]])

def Rot_Z(theta):
    """Rotation matrix for rotation about Z-Axis."""
    return np.array([[np.cos(theta),-np.sin(theta),0.0],
                     [np.sin(theta), np.cos(theta),0.0],
                     [          0.0,           0.0,1.0]])

### Rotation Functions
def Rot_3D(Vector, Alpha, Beta, Gamma):
    """Extrinstic davenport rotation of Vector through euler angles Alpha, Beta, Gamma using z-x-z."""
    R1 = Rot_Z(Alpha)
    R2 = Rot_X(Beta)
    R3 = Rot_Z(Gamma)
    V1 = np.dot(R1,Vector)
    V2 = np.dot(R2,V1)
    Result = np.dot(R3,V2)
    return Result

def Rotate_About_Axis(Vector,Axis,Angle):
    """Rotation of Vector about Axis through Angle."""
    ux = Axis[0]
    uy = Axis[1]
    uz = Axis[2]
    c = np.cos(Angle)
    s = np.sin(Angle)
    R = np.array([[   c+ux*ux*(1-c),ux*uy*(1-c)-uz*s,ux*uz*(1-c)+uy*s],
                  [uy*ux*(1-c)+uz*s,   c+uy*uy*(1-c),uy*uz*(1-c)-ux*s],
                  [uz*ux*(1-c)-uy*s,uz*uy*(1-c)+ux*s,   c+uz*uz*(1-c)]])
    return np.dot(R,Vector)

# Rotation of triangles using Rot_3D
def Rotate_Triangles(Triangles,Alpha,Beta,Gamma):
    """Extrinstic davenport rotation of Triangles through euler angles Alpha, Beta, Gamma using z-x-z."""
    Rotated_Triangles = np.zeros(Triangles.shape)
    for i in range(len(Triangles)):
        Rotated_Triangles[i,0] = Rot_3D(Triangles[i,0], Alpha, Beta, Gamma)
        Rotated_Triangles[i,1] = Rot_3D(Triangles[i,1], Alpha, Beta, Gamma)
        Rotated_Triangles[i,2] = Rot_3D(Triangles[i,2], Alpha, Beta, Gamma)
    return Rotated_Triangles

def Generate_Icosahedron(radius=1.0):
    """Generation of base icosahedron."""
    phi = 0.5*(1.0+np.sqrt(5.0))
    
    Vertices = np.array([[ 0.0, 1.0, phi],
                         [ 0.0, 1.0,-phi],
                         [ 0.0,-1.0, phi],
                         [ 0.0,-1.0,-phi],
                         [ phi, 0.0,-1.0],
                         [ phi, 0.0, 1.0],
                         [-phi, 0.0,-1.0],
                         [-phi, 0.0, 1.0],
                         [ 1.0, phi, 0.0],
                         [ 1.0,-phi, 0.0],
                         [-1.0, phi, 0.0],
                         [-1.0,-phi, 0.0]])*radius/np.linalg.norm([0.0,1.0,phi])
    
    Triangles = np.array([[Vertices[0],Vertices[2],Vertices[5]],
                          [Vertices[0],Vertices[5],Vertices[8]],
                          [Vertices[0],Vertices[8],Vertices[10]],
                          [Vertices[0],Vertices[10],Vertices[7]],
                          [Vertices[0],Vertices[7],Vertices[2]],
                          [Vertices[3],Vertices[1],Vertices[4]],
                          [Vertices[3],Vertices[4],Vertices[9]],
                          [Vertices[3],Vertices[9],Vertices[11]],
                          [Vertices[3],Vertices[11],Vertices[6]],
                          [Vertices[3],Vertices[6],Vertices[1]],
                          [Vertices[1],Vertices[8],Vertices[4]],
                          [Vertices[8],Vertices[5],Vertices[4]],
                          [Vertices[4],Vertices[5],Vertices[9]],
                          [Vertices[5],Vertices[2],Vertices[9]],
                          [Vertices[9],Vertices[2],Vertices[11]],
                          [Vertices[2],Vertices[7],Vertices[11]],
                          [Vertices[11],Vertices[7],Vertices[6]],
                          [Vertices[7],Vertices[10],Vertices[6]],
                          [Vertices[6],Vertices[10],Vertices[1]],
                          [Vertices[10],Vertices[8],Vertices[1]]])
    
    return Triangles

### Functions for determining attributes of spherical triangles
def Get_Triangle_COMs(Triangles):
    """Centres of mass of spherical Triangles."""
    COMs = np.zeros((len(Triangles),3))
    for i in range(len(Triangles)):
        a = np.cross(Triangles[i,0],Triangles[i,1])/np.linalg.norm(np.cross(Triangles[i,0],Triangles[i,1]))
        b = np.cross(Triangles[i,1],Triangles[i,2])/np.linalg.norm(np.cross(Triangles[i,1],Triangles[i,2]))
        c = np.cross(Triangles[i,2],Triangles[i,0])/np.linalg.norm(np.cross(Triangles[i,2],Triangles[i,0]))
        A = np.arccos(np.dot(Triangles[i,0],Triangles[i,1])/(np.linalg.norm(Triangles[i,0])*np.linalg.norm(Triangles[i,1])))
        B = np.arccos(np.dot(Triangles[i,1],Triangles[i,2])/(np.linalg.norm(Triangles[i,1])*np.linalg.norm(Triangles[i,2])))
        C = np.arccos(np.dot(Triangles[i,2],Triangles[i,0])/(np.linalg.norm(Triangles[i,2])*np.linalg.norm(Triangles[i,0])))
        COMs[i] = (a*A+b*B+c*C)/np.linalg.norm(a*A+b*B+c*C)
    return COMs

def Get_Triangle_Base_Areas(Triangles):
    """Areas of spherical Traingles assuming unit sphere."""
    Areas = np.zeros(len(Triangles))
    for i in range(len(Triangles)):
        a = np.arccos(np.dot(Triangles[i,0],Triangles[i,1])/(np.linalg.norm(Triangles[i,0]))*np.linalg.norm(Triangles[i,1]))
        b = np.arccos(np.dot(Triangles[i,1],Triangles[i,2])/(np.linalg.norm(Triangles[i,1]))*np.linalg.norm(Triangles[i,2]))
        c = np.arccos(np.dot(Triangles[i,2],Triangles[i,0])/(np.linalg.norm(Triangles[i,2]))*np.linalg.norm(Triangles[i,0]))
        s = 0.5*(a+b+c)
        Areas[i] = 4.0*np.arctan(np.sqrt(np.tan(0.5*s)*np.tan(0.5*(s-a))*np.tan(0.5*(s-b))*np.tan(0.5*(s-c))))
    return Areas

### Division of triangles
def Divide_Triangles_Angle(Triangles,Amount):
    """Division of triangles into Amount**2 smaller triangles of approximately equal area."""
    A2 = pow(Amount,2)
    Final_Triangles = np.zeros((len(Triangles)*A2,3,3))
    for t in range(len(Triangles)):
        V0 = Triangles[t,0]
        V1 = Triangles[t,1]
        V2 = Triangles[t,2]
        v0 = np.linalg.norm(V0)
        v1 = np.linalg.norm(V1)
        v2 = np.linalg.norm(V2)
        theta0 = np.arccos(np.dot(V0,V1)/(v0*v1))
        theta1 = np.arccos(np.dot(V1,V2)/(v1*v2))
        theta2 = np.arccos(np.dot(V2,V0)/(v2*v0))
        ax0 = np.cross(V0,V1)/np.linalg.norm(np.cross(V0,V1))
        ax1 = np.cross(V1,V2)/np.linalg.norm(np.cross(V1,V2))
        ax2 = np.cross(V2,V0)/np.linalg.norm(np.cross(V2,V0))
        PT = np.zeros((3,Amount+1,Amount+1,3))
        for i in range(Amount+1):
            #0
            PT[0,i,0] = Rotate_About_Axis(V0,ax0,(i/Amount)*theta0)
            PT[0,i,i] = Rotate_About_Axis(V0,-ax2,(i/Amount)*theta2)
            #1
            PT[1,-1,i] = Rotate_About_Axis(V1,ax1,(i/Amount)*theta1)
            PT[1,-1-i,0] = Rotate_About_Axis(V1,-ax0,(i/Amount)*theta0)
            #2
            PT[2,-1-i,-1-i] = Rotate_About_Axis(V2,ax2,(i/Amount)*theta2)
            PT[2,-1,-1-i] = Rotate_About_Axis(V2,-ax1,(i/Amount)*theta1)
            if i>1:
                #0
                Axis_Temp0 = np.cross(PT[0,i,0],PT[0,i,i])/np.linalg.norm(np.cross(PT[0,i,0],PT[0,i,i]))
                Theta_Temp0 = np.arccos(np.dot(PT[0,i,0],PT[0,i,i])/(np.linalg.norm(PT[0,i,0])*np.linalg.norm(PT[0,i,i])))
                #1
                Axis_Temp1 = np.cross(PT[1,-1,i],PT[1,-1-i,0])/np.linalg.norm(np.cross(PT[1,-1,i],PT[1,-1-i,0]))
                Theta_Temp1 = np.arccos(np.dot(PT[1,-1,i],PT[1,-1-i,0])/(np.linalg.norm(PT[1,-1,i])*np.linalg.norm(PT[1,-1-i,0])))
                #2
                Axis_Temp2 = np.cross(PT[2,-1-i,-1-i],PT[2,-1,-1-i])/np.linalg.norm(np.cross(PT[2,-1-i,-1-i],PT[2,-1,-1-i]))
                Theta_Temp2 = np.arccos(np.dot(PT[2,-1-i,-1-i],PT[2,-1,-1-i])/(np.linalg.norm(PT[2,-1-i,-1-i])*np.linalg.norm(PT[2,-1,-1-i])))
                for j in np.arange(1,i):
                    #0
                    PT[0,i,j] = Rotate_About_Axis(PT[0,i,0],Axis_Temp0,(j/i)*Theta_Temp0)
                    #1
                    PT[1,-1-j,i-j] = Rotate_About_Axis(PT[1,-1,i],Axis_Temp1,(j/i)*Theta_Temp1)
                    #2
                    PT[2,-1-i+j,-1-i] = Rotate_About_Axis(PT[2,-1-i,-1-i],Axis_Temp2,(j/i)*Theta_Temp2)
    
        Points = np.zeros((Amount+1,Amount+1,3))
        for i in range(Amount+1):
            for j in range(i+1):
                Points[i,j] = PT[0,i,j] + PT[1,i,j] + PT[2,i,j]
                Points[i,j] = Points[i,j]/np.linalg.norm(Points[i,j])
        for i in range(Amount):
            Final_Triangles[t*A2+i*i,0] = Points[i,0]
            Final_Triangles[t*A2+i*i,1] = Points[i+1,0]
            Final_Triangles[t*A2+i*i,2] = Points[i+1,1]
            for j in range(i):
                Final_Triangles[t*A2+i*i+2*j+1,0] = Points[i,j]
                Final_Triangles[t*A2+i*i+2*j+1,1] = Points[i,j+1]
                Final_Triangles[t*A2+i*i+2*j+1,2] = Points[i+1,j+1]
                
                Final_Triangles[t*A2+i*i+2*j+2,0] = Points[i,j+1]
                Final_Triangles[t*A2+i*i+2*j+2,1] = Points[i+1,j+1]
                Final_Triangles[t*A2+i*i+2*j+2,2] = Points[i+1,j+2]
    return Final_Triangles

def GetSphericalCoordsFromCart(coordsCart):
    """Conversion of coordsCart to spherical angles theta, phi."""
    coordsSpherical = np.zeros((len(coordsCart),2))
    coordsSpherical[:,0] = np.arccos(coordsCart[:,2]/np.linalg.norm(coordsCart,axis=-1))
    coordsSpherical[:,1] = np.arctan2(coordsCart[:,1],coordsCart[:,0])
    #print(np.linalg.norm(coordsCart,axis=-1))
    #print(coordsCart.shape)
    return coordsSpherical

def GetRotationToZ(vector):
    """Rotation required to move vector on to Z-Axis."""
    ZAxis = np.array([0.0,0.0,1.0])
    rotationAxis = np.cross(vector,ZAxis)/np.linalg.norm(np.cross(vector,ZAxis))
    rotationAngle = np.arccos(np.dot(vector,ZAxis)/np.linalg.norm(vector))
    return rotationAxis, -rotationAngle

def RotateVectors(vectors,axis,angle):
    """Wrapper for Rotate_About_Axis."""
    output_vectors = np.zeros(vectors.shape)
    for i in range(len(vectors)):
        output_vectors[i] = Rotate_About_Axis(vectors[i],axis,angle)
    return output_vectors

def GetGeodesicSamplePoints(samples,roundup=True):
    """
    Get samples points based on division of icosahedron faces.

    Args:
    samples - int - approximate number of samples to obtain

    Kwargs:
    roundup=True - bool - if samples != 20*(n^2) then this flag indicates whether to round up or down

    Returns:
    array(float)[samples,2] - theta, phi angles from origin to sample locations
    array(float)[samples] - areas associated with each sample in sr
    """
    division_approx = sqrt(samples/20.0)
    
    if roundup:
        division_exact = int(ceil(division_approx))
    else:
        division_exact = int(floor(division_approx))
    
    icosahedronTriangles = Generate_Icosahedron()
    icosahedronCentresOfMass = Get_Triangle_COMs(icosahedronTriangles)
    
    phi = 0.5*(1.0+np.sqrt(5.0))
    rotationAxis, rotationAngle = GetRotationToZ(np.array([ phi, 0.0, 1.0]))
    
    if division_exact<=1:
        icosahedronCentresOfMass = RotateVectors(icosahedronCentresOfMass,rotationAxis,rotationAngle)
        icosahedronCentresOfMassSpherical = GetSphericalCoordsFromCart(icosahedronCentresOfMass)
        icosahedronAreas = Get_Triangle_Base_Areas(icosahedronTriangles)
        return icosahedronCentresOfMassSpherical, icosahedronAreas
    
    else:
        sampledTriangles = Divide_Triangles_Angle(icosahedronTriangles,division_exact)
        sampledCentresOfMass = Get_Triangle_COMs(sampledTriangles)
        sampledCentresOfMass = RotateVectors(sampledCentresOfMass,rotationAxis,rotationAngle)
        sampledCentresOfMassSpherical = GetSphericalCoordsFromCart(sampledCentresOfMass)
        sampledAreas = Get_Triangle_Base_Areas(sampledTriangles)
        return sampledCentresOfMassSpherical, sampledAreas
    return None


#rad = 1.0
#IcosahedronTriangles = Generate_Icosahedron(rad)
#IcosahedronCentresOfMass = Get_Triangle_COMs(IcosahedronTriangles)
#IcosahedronAreas = Get_Triangle_Base_Areas(IcosahedronTriangles)
#TriO5a = Divide_Triangles_Angle(IcosahedronTriangles,5)