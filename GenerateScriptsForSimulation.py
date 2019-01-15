#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 12:06:16 2018

@author: pch1g13
"""

import datetime
import os
from string import Template

### Setup options ###
# Parent directory of EinsteinToolkitModules
ETMDirectory = "/mainfs/home/pch1g13/Python"

# Scripts subdirectory
scriptDirectory = "Master/Scripts/"

# Simulation Directory
simulationDirectory = "/mainfs/scratch/pch1g13/simulations/nsns_G2_12vs12_long/"

# Substitutions to make in template scripts
substitutions = {"simulationDirectory" : simulationDirectory,
                 "inputFolders" : "128",
                 "dataSubfolder" : "nsns_G2_12vs12_long/",
                 "ETMDirectory" : ETMDirectory,
                 "InitialSeparation" : 67.7,
                 "InitialMass" : 2.4}

# Scripts to create
template_filenames = ["DataMerge_template.py",
                      "GWAnalysis_template.py",
                      "Rho2DFrames_template.py",
                      "ClearCheckpoints_template.py",
                      "Tracer2DFrames_template.py",
                      "VelRho2DFrames_template.py",
                      "SphericalGWAnalysis_template.py",
                      "GWAnalysisWeyl4Extrap_template.py"]

# Names for output scripts
output_filenames = ["DataMerge.py",
                    "GWAnalysis.py",
                    "Rho2DFrames.py",
                    "ClearCheckpoints.py",
                    "Tracer2DFrames.py",
                    "VelRho2DFrames.py",
                    "SphericalGWs.py",
                    "GWAnalysisWeyl4.py"]

### Verify directory structure
if not(os.path.isdir(simulationDirectory + scriptDirectory)):
    print(simulationDirectory + scriptDirectory + " not found, creating")
    os.makedirs(simulationDirectory + scriptDirectory)

if os.listdir(simulationDirectory + scriptDirectory):
    print(simulationDirectory + scriptDirectory + " is not empty.")
    done = False
    while done==False:
        response = input("confirm overwrite? (y/n) ")
        if response=="y":
            done = True
        elif response=="n":
            print("Script generation cancelled.")
            raise RuntimeError("Script destination folder is not empty and overwrite was blocked.")
        else:
            print("Response not understood, respond 'y' to overwite existing scripts, or 'n' to cancel operation.")

### Create scripts
for i in range(len(template_filenames)):
    template_file = open("ScriptTemplates/" + template_filenames[i], "r")
    output_file = open(simulationDirectory + scriptDirectory + output_filenames[i], "w")

    output_file.write('"""\n')
    output_file.write("Created on " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " from " + template_filenames[i] + "\n")
    output_file.write("@author: pch1g13\n")
    output_file.write('"""\n')
                  
    for line in template_file:
        output_file.write(Template(line).substitute(substitutions))

    output_file.close()
    template_file.close()