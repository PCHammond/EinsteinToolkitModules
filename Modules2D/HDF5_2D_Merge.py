#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 16:47:34 2018

@author: pch1g13

Functions for merging 2+1-dimensional data in HDF5 files.
"""
import sys, os.path
ETM_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(ETM_dir)

import h5py
import numpy as np
from subprocess import run
from Common import GetIterationFromKey

def MergeHDF52D(simulation_directory, 
                input_directories, 
                input_filename, 
                output_directory, 
                output_filename, 
                verbose=False,
                overwrite_files=False, 
                skip_if_exists=False):
    """
    Merge together HDF5 files in different directories.

    Args:
    simulation_directory - str - absolute directory of simulation data
    input_directories - list(str) - relative directorys to HDF5 files to be merged from simulation directory
    input_filename - str - name of HDF5 file to be merged
    output_directory - str - location to save merged HDF5 data. Taken as relative from simulation_directory
    output_filename - str - filename for output HDF5 file

    Kwargs:
    verbose=False - bool - verbosity of output to IO
    overwrite_files=False - bool - overwrite existing output files if present
    skip_if_exists=False - bool - abort merging if output file already exists
    
    This function is intended for use in zipping together files output by the 
    Einstein Toolkit. The different directories in 'input_directories' are read
    in reverse order, so that given a list of directories in chronlogically 
    produced order, if there are duplicates of datapoints, those in newer files
    are preferred.
    
    Data is read from /simulation_directory/input_directories/input_filename 
    and saved to /simulation_directory/output_directory/output_filename.
    """
    iterations_copied = []
    iterations_checked = []

    if overwrite_files and skip_if_exists:
        raise ValueError("Cannot skip and overwrite")
    
    if os.path.isfile(simulation_directory + output_directory + output_filename):
        if skip_if_exists:
            if verbose: print(simulation_directory + output_directory + output_filename + " already exists, skipping")
            return None
        elif overwrite_files:
            if verbose: print(simulation_directory + output_directory + output_filename + " already exists, overwriting")
            run(["rm",simulation_directory + output_directory + output_filename])
        else:
            raise ValueError("Output file exists, please run with skipping or overwriting enabled")     
    
    file_output = h5py.File(simulation_directory + output_directory + output_filename,"x")

    if verbose: print("Merging " + input_filename)
    for dir_data in input_directories[::-1]:
        file_current = h5py.File(simulation_directory + dir_data + input_filename,"r")
        keyList_current = list(file_current.keys())
        if verbose: print(str(len(keyList_current)) + " keys in " + dir_data)
        iterations_current = []
        for key in keyList_current:
            if type(file_current[key])==h5py._hl.dataset.Dataset:
                iteration = GetIterationFromKey(key)
                iterations_checked.append(iteration)
                if not(iteration in iterations_copied):
                    file_current.copy(key,file_output)
                if not(iteration in iterations_current):
                    iterations_current.append(iteration)
                    
        iterations_copied += iterations_current
    
    file_current.copy('Parameters and Global Attributes',file_output)
    file_current.close()
    file_output.close()
    
    if verbose: print(output_directory + output_filename + " written")
    
    return None