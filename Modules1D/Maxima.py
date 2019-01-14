#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 18:33:59 2018

@author: pch1g13

Functions for determining variable maxima values and locations.
"""

import numpy as np

def FindMaxRho(simulation_directory,input_directory):
    """Get location of maximum value of density."""
    data_raw = np.load(simulation_directory + input_directory + "hydrobase-rho.scalars.npy")
    return np.max(data_raw[:,3])
    