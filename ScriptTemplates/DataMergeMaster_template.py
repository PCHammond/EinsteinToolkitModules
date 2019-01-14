### Add Python folder to sys.path ###
import sys
sys.path.append("${ETMDirectory}")

import os
from subprocess import run
from EinsteinToolkitModules.Common import GenerateDirectories
from EinsteinToolkitModules.Modules1D.ASC_1D_Merge import MergeASCFilesToNPY, GenerateMultipolePsi4Names
from EinsteinToolkitModules.Modules2D.HDF5_2D_Merge import MergeHDF52D
from EinsteinToolkitModules.Modules3D.HDF5_3D_Merge import MergeHDF53D, GenerateSplitFilenames

### Input data directories and options ###
simulation_directory = "${simulationDirectory}"
input_directories = GenerateDirectories(${inputFolders},"${dataSubfolder}")

# Verbosity
verbose = True
veryverbose = False

# Options for overwriting files
skip_if_exists = False
overwrite_files = True
create_output_dirs = True

### ASC Data ###
input_filenames_asc = ["hydro_analysis-hydro_analysis_rho_max_loc..asc",
                       "hydro_analysis-hydro_analysis_rho_max_origin_distance..asc",
                       "hydrobase-rho.scalars.asc"]

output_directory_asc = "Master/ASC/"

output_filenames_asc = ["hydro_analysis-hydro_analysis_rho_max_loc",
                        "hydro_analysis-hydro_analysis_rho_max_origin_distance",
                        "hydrobase-rho.scalars"]

radii_psi4 = ["45.00",
              "70.00",
              "100.00",
              "125.00",
              "150.00",
              "200.00",
              "250.00",
              "300.00",
              "350.00",
              "400.00",
              "500.00",
              "600.00",
              "700.00",
              "800.00",
              "900.00",
              "1000.00"]

multipoles_psi4 = 8

input_filenames_psi4 = GenerateMultipolePsi4Names(radii_psi4,multipoles_psi4)

output_filenames_psi4 = input_filenames_psi4

### HDF5 2D Data ###
input_filenames_2D = ["alp.xy.h5",
                      "betax.xy.h5",
                      "betay.xy.h5",
                      "betaz.xy.h5",
                      "cons_tracer[0].xy.h5",
                      "dens.xy.h5",
                      "eps.xy.h5",
                      "press.xy.h5",
                      "rho.xy.h5",
                      "tracer[0].xy.h5",
                      "vel[0].xy.h5",
                      "vel[1].xy.h5",
                      "vel[2].xy.h5",
                      "w_lorentz.xy.h5"]

output_directory_2D = "Master/HDF5/"

output_filenames_2D = input_filenames_2D

### HDF5 3D Data ###
variables_3D = ["rho.xyz",
                "eps.xyz",
                "dens.xyz",
                "press.xyz",
                "w_lorentz.xyz",
                "vel[0].xyz",
                "vel[1].xyz",
                "vel[2].xyz",
                "cons_tracer[0].xyz",
                "tracer[0].xyz"]

input_filenames_3D = []
output_filenames_3D = []

for var in variables_3D:
    input_filenames_3D.append(GenerateSplitFilenames(var,4))
    output_filenames_3D.append(var + ".h5")

output_directory_3D = "Master/HDF5/"

### Check Directory Structure ###
if not(os.path.isdir(simulation_directory)):
    print(simulation_directory + " not found, stopping")
    raise ValueError

for in_dir in input_directories:
    if not(os.path.isdir(simulation_directory + in_dir)):
        print(simulation_directory + in_dir + " not found, stopping")
        raise ValueError

if not(os.path.isdir(simulation_directory + output_directory_asc)):
    if create_output_dirs:
        print(simulation_directory + output_directory_asc + " not found, creating")
        os.makedirs(simulation_directory + output_directory_asc)
    else:
        print(simulation_directory + output_directory_asc + " not found, stopping")
        raise ValueError
    
if not(os.path.isdir(simulation_directory + output_directory_2D)):
    if create_output_dirs:
        print(simulation_directory + output_directory_2D + " not found, creating")
        os.makedirs(simulation_directory + output_directory_2D)
    else:
        print(simulation_directory + output_directory_2D + " not found, stopping")
        raise ValueError
    
if not(os.path.isdir(simulation_directory + output_directory_3D)):
    if create_output_dirs:
        print(simulation_directory + output_directory_3D + " not found, creating")
        os.makedirs(simulation_directory + output_directory_3D)
    else:
        print(simulation_directory + output_directory_3D + " not found, stopping")
        raise ValueError

### Do ASC Merge ###
for i in range(len(input_filenames_asc)):
    if verbose or veryverbose: print("Merging " + input_filenames_asc[i])
    MergeASCFilesToNPY(simulation_directory, 
                       input_directories, 
                       input_filenames_asc[i], 
                       output_directory=output_directory_asc, 
                       output_filename=output_filenames_asc[i], 
                       verbose=veryverbose,
                       remove_asc=True,
                       overwrite_files=overwrite_files,
                       skip_if_exists=skip_if_exists)

for i in range(len(input_filenames_psi4)):
    if verbose or veryverbose: print("Merging " + input_filenames_psi4[i])
    MergeASCFilesToNPY(simulation_directory, 
                       input_directories, 
                       input_filenames_psi4[i], 
                       output_directory=output_directory_asc, 
                       output_filename=output_filenames_psi4[i], 
                       verbose=veryverbose,
                       remove_asc=True,
                       overwrite_files=overwrite_files,
                       skip_if_exists=skip_if_exists)

### Do HDF5 2D Merge ###
for i in range(len(input_filenames_2D)):
    MergeHDF52D(simulation_directory, 
                input_directories, 
                input_filenames_2D[i], 
                output_directory_2D, 
                output_filenames_2D[i], 
                verbose=verbose,
                overwrite_files=overwrite_files,
                skip_if_exists=skip_if_exists)
    
### Do HDF5 3D Merge ###
for i in range(len(output_filenames_3D)):
    MergeHDF53D(simulation_directory, 
                input_directories, 
                input_filenames_3D[i], 
                output_directory_3D, 
                output_filenames_3D[i], 
                verbose=verbose, 
                overwrite_files=overwrite_files,
                skip_if_exists=skip_if_exists)
