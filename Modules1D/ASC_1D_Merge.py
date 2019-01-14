#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 12:34:05 2018

@author: pete

Functions for merging 1+1-dimensional data in ASC files.
"""

import numpy as np
import os
from subprocess import run

def MergeASCFilesToNPY(simulation_directory, 
                       input_directories, 
                       input_filename, 
                       output_directory=None, 
                       output_filename=None, 
                       remove_asc=False, 
                       verbose=False, 
                       overwrite_files=False, 
                       skip_if_exists=False):
    """
    Merge together .asc files in different directories.

    Args:
    simulation_directory - str - absolute directory of simulation data
    input_directories - list(str) - relative directorys to ASC files to be merged from simulation directory
    input_filename - str - name of ASC file to be merged

    Kwargs:
    output_directory=None - str - option location to save merged ASC data as .npy. Taken as relative from simulation_directory
    output_filename=None - str - filename for output .npy
    remove_asc=False - bool - remove .asc from end of filename when saving as .npy
    verbose=False - bool - verbosity of output to IO
    overwrite_files=False - bool - overwrite existing output files if present
    skip_if_exists=False - bool - abort merging if output file already exists

    Returns:
    array - merged data
    
    This function is intended for use in zipping together files output by the 
    Einstein Toolkit. The different directories in 'input_directories' are read
    in reverse order, so that given a list of directories in chronlogically 
    produced order, if there are duplicates of datapoints, those in newer files
    are preferred.
    
    Data is read from /simulation_directory/input_directories/input_filename 
    and saved to /simulation_directory/output_directory/output_filename.npy if 
    and only if both output_directory and output_filename are given.
    """
    if output_directory and output_filename:
        if overwrite_files and skip_if_exists:
            raise ValueError("Cannot skip and overwrite")

        if output_filename[-4:]==".asc":
            if remove_asc:
                output_filename_temp = output_filename[:-4]
            else:
                output_filename_temp = output_filename
        else:
            output_filename_temp = output_filename
        
        if os.path.isfile(simulation_directory + output_directory + output_filename_temp + ".npy"):
            if skip_if_exists:
                if verbose: print(simulation_directory + output_directory + output_filename_temp + ".npy" + " already exists, skipping")
                return None
            elif overwrite_files:
                if verbose: print(simulation_directory + output_directory + output_filename_temp + ".npy" + " already exists, overwriting")
                run(["rm",simulation_directory + output_directory + output_filename_temp + ".npy"])
            else:
                raise ValueError("Output file exists, please run with skipping or overwriting enabled")              
    
    data_output_list = []
    times_copied = []
    
    for data_dir in input_directories[::-1]:
        if verbose: print("Opening " + input_filename + " in " + data_dir)
        data_current = np.genfromtxt(simulation_directory + data_dir + input_filename)
        for time_idx in range(len(data_current)):
            time_current = data_current[time_idx,0]
            if not(time_current in times_copied):
                data_output_list.append(data_current[time_idx])
                times_copied.append(time_current)
    data_output_unsorted = np.asarray(data_output_list)
    data_output = data_output_unsorted[np.argsort(data_output_unsorted[:,0])]
    
    if output_directory and output_filename_temp:
        np.save(simulation_directory + output_directory + output_filename_temp + ".npy", data_output)
    return data_output

def GenerateMultipolePsi4Names(radii,max_harmonic):
    """
    Generate names of Weyl 4 multipole ASC files for list of harmonics and radii.

    Args:
    radii - list(str) - radii for which to generate filenames
    max_harmonic - int - maximum spherical harmonic for which to generate filenames

    Returns:
    list(str) - filenames of form mp_Psi4_l{l}_m{m}_r{radius}.asc for desired radii and harmonics
    """
    harmonics = []
    filenames = []

    for l in range(max_harmonic+1):
        for m in range(-l,l+1):
            harmonics.append([l,m])
    
    for lm in harmonics:
        for rad in radii:
            filenames.append("mp_Psi4_l" + str(lm[0]) + "_m" + str(lm[1]) + "_r" + rad + ".asc")
    
    return filenames