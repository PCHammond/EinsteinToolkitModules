#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 12:34:05 2018

@author: pete

Common functions for EinsteinToolkitModules
"""

import numpy as np
from math import sqrt, sin, tan, factorial, pi
from cmath import exp
from scipy.special import comb, expit

# Constants for unit conversion
c = 299792458 # m s-1
G = 6.67408e-11 # m3 Kg-1 s-2
Mdot = 1.98847e30 # Kg

def GetMassFactor(output_units):
    """
    Unit conversion from Geometric Units for mass.
    
    Args:
    output_units - str - unit system to convert into

    Returns:
    float - conversion factor from Geometric to chosen units

    Supported values of output_units are:
    "SI" - convert to kilograms
    "cgs" - convert to grams

    Usage:
    To convert X in geometric units to Y in units y: Y = X * GetMassFactor("y")
    """
    if output_units=="SI":
        return Mdot
    elif output_units=="cgs":
        return Mdot*1000.0
    else:
        raise ValueError("output_units specified not recognised. See DocString for available units.")

def GetLengthFactor(output_units):
    """
    Unit conversion from Geometric Units for length.
    
    Args:
    output_units - str - unit system to convert into

    Returns:
    float - conversion factor from Geometric to chosen units

    Supported values of output_units are:
    "SI" - convert to meters
    "cm" - convert to centimeters
    "km" - convert to kilometers

    Depreciated:
    "cgs" - convert to kilometers

    Usage:
    To convert X in geometric units to Y in units y: Y = X * GetLengthFactor("y")
    """
    if output_units=="SI":
        return G*Mdot*c**-2
    elif output_units=="cgs":
        print('Option "cgs" is depreciated, use either "cm" for centimeters, or "km" for kilometers. "cgs" returns "km".')
        return (G*Mdot*c**-2)/1000.0
    elif output_units=="cm":
        return (G*Mdot*c**-2)*100.0
    elif output_units=="km":
        return (G*Mdot*c**-2)/1000.0
    else:
        raise ValueError("output_units specified not recognised. See DocString for available units.")

def GetTimeFactor(output_units):
    """
    Unit conversion from Geometric Units for time.
    
    Args:
    output_units - str - unit system to convert into

    Returns:
    float - conversion factor from Geometric to chosen units

    Supported values of output_units are:
    "SI" - convert to seconds
    "cgs" - convert to seconds
    "ms" - convert to milliseconds
    "us" - convert to microseconds

    Usage:
    To convert X in geometric units to Y in units y: Y = X * GetLengthFactor("y")
    """
    if output_units=="SI":
        return G*Mdot*c**-3
    elif output_units=="cgs":
        return G*Mdot*c**-3
    elif output_units=="us":
        return (G*Mdot*c**-3)*1.0e6
    elif output_units=="ms":
        return (G*Mdot*c**-3)*1.0e3
    else:
        raise ValueError("output_units specified not recognised. See DocString for available units.")

def GenerateDirectories(number_of_directories, subfolder=None):
    """
    Generate list of output directory strings, with optional subfolder.

    Args:
    number_of_directories - int - number of output directory folders required

    Kwargs:
    subfolder=None - str - subdirectory string inside each output directory to append to the end

    Returns:
    list - List containging output directory strings with optional subfolder in format "output-xxxx/{optional/sub/directory/}"
    """
    data_directories_list = []
    for i in range(number_of_directories):
        data_directories_list.append("output-" + str(i).zfill(4) + "/")
        if subfolder:
            data_directories_list[-1] += subfolder
            if subfolder[-1] != "/":
                data_directories_list[-1] += "/"
    return data_directories_list

def GetIterationFromKey(key):
    """
    Get iteration number for given HDF5 key.

    Args:
    key - str - HDF5 key containing "it={iteration}"

    Returns:
    int - iteration number extracted from string.
    """
    try:
        index_current = key.index("it=") + 3
    except:
        return -1
    done = False
    output_string = ""
    while done == False:
        output_string += key[index_current]
        index_current += 1
        if key[index_current] == " ":
            done = True
        
    return int(output_string)

def GetRefinementLevelFromKey(key):
    """
    Get refinement level for given HDF5 key.

    Args:
    key - str - HDF5 key containing "rl={refinementlevel}"

    Returns:
    int - refinement level extracted from string.
    """
    try:
        index_current = key.index("rl=") + 3
    except:
        return -1
    done = False
    output_string = ""
    while done == False:
        output_string += key[index_current]
        index_current += 1
        if key[index_current] == " ":
            done = True
        
    return int(output_string)

def GetIterationArray(list_keys):
    """
    Get unique iteration numbers from given list of HDF5 keys.

    Args:
    list_keys - list(str) - list of HDF5 keys containing "it={iteration}"

    Returns:
    array(int) - sorted array containing unique iteration numbers in given list
    """
    list_iterations = []
    
    for key in list_keys:
        iteration_current = GetIterationFromKey(key)
        if (iteration_current>=0) and (not(iteration_current in list_iterations)):
            list_iterations.append(iteration_current)
    
    return np.sort(np.asarray(list_iterations))

def GetRefinementLevelArray(list_keys):
    """
    Get unique refinement levels from given list of HDF5 keys.

    Args:
    list_keys - list(str) - list of HDF5 keys containing "rl={refinementlevel}"

    Returns:
    array(int) - sorted array containing unique refinement levels in given list
    """
    list_rLs = []
    
    for key in list_keys:
        rL_current = GetRefinementLevelFromKey(key)
        if (rL_current>=0) and (not(rL_current in list_rLs)):
            list_rLs.append(rL_current)
    
    return np.sort(np.asarray(list_rLs))

def GetKeysForIteration(iteration,list_keys):
    """
    Extract subset of keys from list for given iteration.

    Args:
    iteration - int - target iteration number
    list_keys - list[str] - list of HDF5 keys containing "it={iteration}"

    Returns:
    list(str) - list of keys that match iteration number given
    """
    string_target = "it=" + str(iteration) + " "
    list_keys_output = []
    for key in list_keys:
        if string_target in key:
            list_keys_output.append(key)
    return list_keys_output

def GetKeysForRefinement_Level(rL,list_keys):
    """
    Extract subset of keys from list for given refinement level.

    Args:
    rL - int - target refinement level
    list_keys - list[str] - list of HDF5 keys containing "rl={refinementlevel}"

    Returns:
    list(str) - list of keys that match refinement level given
    """
    string_target = "rl=" + str(rL) + " "
    list_keys_output = []
    for key in list_keys:
        if string_target in key:
            list_keys_output.append(key)
    return list_keys_output

def LogisticFunction(x,k,L=1.0,x0=0.0):
    """
    Get logisitc funtion of x with steepness k.

    Args
    x - array(float) - domain of logistic function
    k - float - steepness

    Kwargs:
    L=1.0 - float - height of step
    x0=0.0 - float - location of midpoint
    """
    return L*expit(k*(x-x0))