### Add Python folder to sys.path ###
import sys
sys.path.append("${ETMDirectory}")

### Load modules for analysis
import os
import h5py
import numpy as np
from EinsteinToolkitModules.Modules2D.Interp2D import Interp2DScalar, Interp2DSpeedDirection
from EinsteinToolkitModules.Modules2D.Common2D import GetSmallestCoveringRL2D, GetDomainAfterTransforms
from EinsteinToolkitModules.Modules2D.Plot2D import PlotVelRhoRGB2D
from EinsteinToolkitModules.Common import GetIterationArray, GetRefinementLevelArray, GetKeysForIteration
from EinsteinToolkitModules.Common import GetLengthFactor, GetTimeFactor, GetMassFactor
from EinsteinToolkitModules.Modules1D.Maxima import FindMaxRho

### Directory setup
# Input
simulation_directory = "${simulationDirectory}"
input_directory = "Master/HDF5/"
file_name_rho = "rho.xy.h5"
file_name_velx = "vel[0].xy.h5"
file_name_vely = "vel[1].xy.h5"

# Output
output_directory = "Master/Videos/VelRho/Frames/"
file_name_o = "velrho.xy"

### Bounds, resolution, and transformations for data
bounds_output = np.array([0.0,72.0,-72.0,72.0]) ##xmin, xmax, ymin, ymax
samples_output = np.array([1025,2049]) ##num_x, num_y
transforms = ["rotate_right_half"]

### Bounds for value of data
Rho_max = FindMaxRho(simulation_directory,"Master/ASC/")
Rho_min = 1e-11

Speed_max = 0.5

### Create output directory if absent
if not(os.path.isdir(simulation_directory + output_directory)):
    print(simulation_directory + output_directory + " not found, creating")
    os.makedirs(simulation_directory + output_directory)

### Load data
# Open data files
file_input_rho = h5py.File(simulation_directory + input_directory + file_name_rho,'r')
file_input_velx = h5py.File(simulation_directory + input_directory + file_name_velx,'r')
file_input_vely = h5py.File(simulation_directory + input_directory + file_name_vely,'r')

# Get lists of keys in HDF5 files
list_keys_rho = list(file_input_rho.keys())
list_keys_velx = list(file_input_velx.keys())
list_keys_vely = list(file_input_vely.keys())

# Get lists of iterations present in data files
list_iterations_rho = GetIterationArray(list_keys_rho)
list_iterations_velx = GetIterationArray(list_keys_velx)
list_iterations_vely = GetIterationArray(list_keys_vely)

### Set up domain
# Spacial domain
xs_output = np.linspace(bounds_output[0],bounds_output[1],num=samples_output[0])
ys_output = np.linspace(bounds_output[2],bounds_output[3],num=samples_output[1])

# Options for time domain
frame_offset = 0
#list_iterations = [0]
#list_iterations = [list_iterations[1],list_iterations[-1]]
#list_iterations = list_iterations[frame_offset:]

### Unit conversion from geometric to cgs/km
L_cm = GetLengthFactor("cm")
L_km = GetLengthFactor("km")
T_ms = GetTimeFactor("ms")
T_us = GetTimeFactor("us")
M_ = GetMassFactor("cgs")
Rho_cgs = M_/(L_cm**3)

xs_plot, ys_plot = GetDomainAfterTransforms(xs_output,ys_output,transforms)

### Plotting options
# Options passed to imshow
imshow_kwargs = {"origin":"lower",
                 "interpolation":"bessel"}

# Options for plot labels
plot_labels = {"Title":r"Velocity, Density",
               "xAxis":r"$$x \left[ \mathrm{km} \right]$$",
               "yAxis":r"$$y \left[ \mathrm{km} \right]$$",
               "Colourbar":r"$$\rho \left[ \frac{\mathrm{g}}{\mathrm{cm}^3} \right]$$"}

# Data bounds
data_kwargs = {"log_rho_min" : np.log10(Rho_min*Rho_cgs),
               "log_rho_max" : np.log10(Rho_max*Rho_cgs),
               "speed_min" : 0.0,
               "speed_max" : Speed_max}

### Anaylsis loop
for it_idx in range(len(list_iterations_rho)):
    # Get current iteration
    iteration_current = list_iterations_rho[it_idx]

    # Get lists of HDF5 keys that match current iteration
    list_keys_iteration_rho = GetKeysForIteration(iteration_current,list_keys_rho)
    list_keys_iteration_velx = GetKeysForIteration(iteration_current,list_keys_velx)
    list_keys_iteration_vely = GetKeysForIteration(iteration_current,list_keys_vely)
    
    # Determine current time
    time_current = file_input_rho[list_keys_iteration_rho[0]].attrs['time']
    
    # Set time label for plot
    plot_labels["Time"] = r"$$t = " + r"{:.3f}".format(time_current*T_ms) + r" \mathrm{ms}$$"
    
    # Set filename for current figure
    filename_current = file_name_o + ".fig" + str(it_idx+frame_offset).zfill(4)
    
    # Print info to IO
    print("Creating " + filename_current + " for iteration " + str(iteration_current) + ", time = " + str("{0:.3f}".format(time_current)))
    
    # Get list of refinement levels for current iteration
    list_rLs_current = GetRefinementLevelArray(list_keys_iteration_rho)
    
    # Determine coarsest refinement level that covers domain
    rL_start = GetSmallestCoveringRL2D(list_rLs_current,list_keys_iteration_rho,xs_output,ys_output,file_input_rho)
    
    # Choose coarsest refinement level required and finer
    interp_rLs = list_rLs_current[np.nonzero(list_rLs_current == rL_start)[0][0]:]
    
    # Do interpolation of density for current iteration
    interped_data_rho = Interp2DScalar(interp_rLs,
                                       list_keys_iteration_rho,
                                       xs_output,
                                       ys_output,
                                       file_input_rho,
                                       do_clip = True, 
                                       clip_bounds = np.array([Rho_min,Rho_max]), 
                                       k_spline = 1,
                                       transforms = transforms) * Rho_cgs
    
    # Do interpolation of fluid speed and direction for current iteration
    interped_data_speed, interped_data_direc = Interp2DSpeedDirection(interp_rLs,
                                                                      list_keys_iteration_velx,
                                                                      list_keys_iteration_vely,
                                                                      xs_output,
                                                                      ys_output,
                                                                      file_input_velx,
                                                                      file_input_vely,
                                                                      do_clip=[True,True],
                                                                      clip_bounds=np.array([[data_kwargs["speed_min"],data_kwargs["speed_max"]],[-np.pi,np.pi]]),
                                                                      k_spline=1,
                                                                      transforms = transforms)
    
    # Create and save figure for current iteration
    PlotVelRhoRGB2D(interped_data_speed,
                    interped_data_direc,
                    np.log10(interped_data_rho),
                    xs_plot,
                    ys_plot,
                    simulation_directory,
                    output_directory,
                    filename_current,
                    data_kwargs=data_kwargs,
                    fig_size=[8,4.5],
                    fig_dpi=480,
                    imshow_kwargs=imshow_kwargs,
                    labels=plot_labels,
                    verbose=False)