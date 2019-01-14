#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 16:35:05 2018

@author: pch1g13

Functions for dealing with spin-weighted spherical harmonics.
"""

from scipy.special import comb
from math import sqrt, factorial, pi, sin, cos
from cmath import exp

### _s Y _l _m
def GetSpinWeightedSphericalYDD(s,l,m,theta,phi):
    """
    Get sYlm(theta,phi)

    Args:
    s - int - spin weight
    l - int - degree
    m - int - order
    theta - float - polar angle
    phi - float - azimuth

    Returns:
    complex - sYlm(theta,phi)
    """
    if theta==0:
        if m != 2:
            return 0.0 + 0.0j
        else:
            return sqrt((2*l+1)/(4*pi))*exp(2.0j*phi)
    elif theta==pi:
        if m!=-2:
            return 0.0 + 0.0j
        else: 
            return sqrt((2*l+1)/(4*pi))*exp(2.0j*phi)
        
    part1 = ((-1)**(m)) * exp(1.0j*m*phi) * sqrt((factorial(l+m)*factorial(l-m)*(2*l+1))/
                               (4*pi*factorial(l+s)*factorial(l-s)))
    part2 = 0.0 + 0.0j
    for r in range(0,l-s+1):
        if (r<=l-s)and(r+s-m<=l+s):
            part2 += (((sin(0.5*theta)**(2*l+m-2*r-s))*(cos(0.5*theta)**(2*r-m+s))) * 
                      comb(l-s,r) * 
                      comb(l+s,r+s-m) * 
                      ((-1)**(l-r-s)))
    return part1 * part2

### _s Y ^l _m
def GetSpinWeightedSphericalYUD(s,l,m,theta,phi):
    if m==0:
        return GetSpinWeightedSphericalYDD(s,l,m,theta,phi)
    elif m<0:
        return (1.0/sqrt(2))*(GetSpinWeightedSphericalYDD(s,l,-m,theta,phi)-
                              1.0j*GetSpinWeightedSphericalYDD(s,l,-m,theta,phi))
    elif m>0:
        return (((-1)**m)/sqrt(2))*(GetSpinWeightedSphericalYDD(s,l,m,theta,phi)+
                                    1.0j*GetSpinWeightedSphericalYDD(s,l,-m,theta,phi))