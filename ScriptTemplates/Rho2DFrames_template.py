### Add Python folder to sys.path ###
import sys
sys.path.append("${ETMDirectory}")

### Load modules for analysis
import os
import h5py
import numpy as np
from itertools import zip_longest, repeat
from multiprocessing import Pool
from EinsteinToolkitModules.Modules2D.Interp2D import Interp2DScalar
from EinsteinToolkitModules.Modules2D.Common2D import GetSmallestCoveringRL2D, GetDomainAfterTransforms
from EinsteinToolkitModules.Modules2D.Plot2D import PlotScalarLog2D
from EinsteinToolkitModules.Common import GetIterationArray, GetRefinementLevelArray, GetKeysForIteration
from EinsteinToolkitModules.Common import GetLengthFactor, GetTimeFactor, GetMassFactor
from EinsteinToolkitModules.Modules1D.Maxima import FindMaxRho

### Wrapper function for producing figure
def Make_Figure(it_idx,
                list_keys,
                plot_labels,
                T_,
                file_name_o,
                frame_offset,
                xs_output,
                ys_output,
                file_input,
                transforms,
                Rho_cgs,
                xs_plot,
                ys_plot,
                simulation_directory,
                output_directory,
                imshow_kwargs):
    iteration_current = list_iterations[it_idx]
    list_keys_iteration = GetKeysForIteration(iteration_current,list_keys)
    time_current = file_input[list_keys_iteration[0]].attrs['time']
    plot_labels["Time"] = r"$$t = " + r"{:.3f}".format(time_current*T_) + r" \mathrm{ms}$$"
    filename_current = file_name_o + ".fig" + str(it_idx+frame_offset).zfill(4)
    print("Creating " + filename_current + " for iteration " + str(iteration_current) + ", time = " + str("{0:.3f}".format(time_current)))
    list_rLs_current = GetRefinementLevelArray(list_keys_iteration)
    rL_start = GetSmallestCoveringRL2D(list_rLs_current,list_keys_iteration,xs_output,ys_output,file_input)
    interp_rLs = list_rLs_current[np.nonzero(list_rLs_current == rL_start)[0][0]:]
    interped_data = Interp2DScalar(interp_rLs,
                                   list_keys_iteration,
                                   xs_output,
                                   ys_output,
                                   file_input,
                                   do_clip = True, 
                                   clip_bounds = np.array([1e-11,1]), 
                                   k_spline = 1,
                                   transforms = transforms) * Rho_cgs
    PlotScalarLog2D(interped_data,
                    xs_plot,
                    ys_plot,
                    simulation_directory,
                    output_directory,
                    filename_current,
                    fig_size=[8,4.5],
                    fig_dpi=480,
                    imshow_kwargs=imshow_kwargs,
                    labels=plot_labels,
                    verbose=False)
    return iteration_current

### Directory setup
# Input
simulation_directory = "${simulationDirectory}"
input_directory = "Master/HDF5/"
file_name = "rho.xy.h5"

# Output
output_directory = "Master/Videos/Density/Frames/"
file_name_o = "rho.xy"

### Bounds, resolution, and transformations for data
bounds_output = np.array([0.0,72.0,-72.0,72.0]) ##xmin, xmax, ymin, ymax
samples_output = np.array([1025,2049]) ##num_x, num_y
transforms = ["rotate_right_half"]

### Bounds for value of density
Rho_max = FindMaxRho(simulation_directory,"Master/ASC/")
Rho_min = 1e-11

### Create output directory if absent
if not(os.path.isdir(simulation_directory + output_directory)):
    print(simulation_directory + output_directory + " not found, creating")
    os.makedirs(simulation_directory + output_directory)

### Load data
# Open data file
file_input = h5py.File(simulation_directory + input_directory + file_name,'r')

# Get list of keys in HDF5 file
list_keys = list(file_input.keys())

# Get list of iterations present in data file
list_iterations = GetIterationArray(list_keys)

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

Rho_max_cgs = Rho_max*Rho_cgs
Rho_min_cgs = Rho_min*Rho_cgs
log_Rho_max_cgs = np.log10(Rho_max_cgs)
log_Rho_min_cgs = np.log10(Rho_min_cgs)
xs_plot, ys_plot = GetDomainAfterTransforms(xs_output,ys_output,transforms)

### Plotting options
# Options passed to imshow
imshow_kwargs = {"origin":"lower",
                 "vmin":log_Rho_min_cgs,
                 "vmax":log_Rho_max_cgs,
                 "interpolation":"bessel",
                 "cmap":"inferno"}

# Options for plot labels
plot_labels = {"Title":r"Density",
               "xAxis":r"$$x \left[ \mathrm{km} \right]$$",
               "yAxis":r"$$y \left[ \mathrm{km} \right]$$",
               "Colourbar":r"$$\rho \left[ \frac{\mathrm{g}}{\mathrm{cm}^3} \right]$$"}

### Lists for parallelisation
core_count = 4
iteration_count = len(list_iterations)
args = zip_longest(range(iteration_count),
                   repeat(list_keys,iteration_count),
                   repeat(plot_labels,iteration_count),
                   repeat(T_ms,iteration_count),
                   repeat(file_name_o,iteration_count),
                   repeat(frame_offset,iteration_count),
                   repeat(xs_output,iteration_count),
                   repeat(ys_output,iteration_count),
                   repeat(file_input,iteration_count),
                   repeat(transforms,iteration_count),
                   repeat(Rho_cgs,iteration_count),
                   repeat(xs_plot,iteration_count),
                   repeat(ys_plot,iteration_count),
                   repeat(simulation_directory,iteration_count),
                   repeat(output_directory,iteration_count),
                   repeat(imshow_kwargs,iteration_count))

### Do parallel figures
pool = Pool(core_count)
iterations_plotted = pool.starmap(Make_Figure, args)

### Post processing


################
### Old Code ###
################

"""

### Anaylsis loop
for it_idx in range(iteration_count):
    # Get current iteration
    iteration_current = list_iterations[it_idx]

    # Get list of HDF5 keys that match current iteration
    list_keys_iteration = GetKeysForIteration(iteration_current,list_keys)

    # Determine current time
    time_current = file_input[list_keys_iteration[0]].attrs['time']

    # Set time label for plot
    plot_labels["Time"] = r"$$t = " + r"{:.3f}".format(time_current*T_ms) + r" \mathrm{ms}$$"

    # Set filename for current figure
    filename_current = file_name_o + ".fig" + str(it_idx+frame_offset).zfill(4)

    # Print info to IO
    print("Creating " + filename_current + " for iteration " + str(iteration_current) + ", time = " + str("{0:.3f}".format(time_current)))

    # Get list of refinement levels for current iteration
    list_rLs_current = GetRefinementLevelArray(list_keys_iteration)

    # Determine coarsest refinement level that covers domain
    rL_start = GetSmallestCoveringRL2D(list_rLs_current,list_keys_iteration,xs_output,ys_output,file_input)

    # Choose coarsest refinement level required and finer
    interp_rLs = list_rLs_current[np.nonzero(list_rLs_current == rL_start)[0][0]:]

    # Do interpolation of density for current iteration
    interped_data = Interp2DScalar(interp_rLs,
                                   list_keys_iteration,
                                   xs_output,
                                   ys_output,
                                   file_input,
                                   do_clip = True, 
                                   clip_bounds = np.array([1e-11,1]), 
                                   k_spline = 1,
                                   transforms = transforms) * Rho_cgs
    
    # Create and save figure for current iteration
    PlotScalarLog2D(interped_data,
                    xs_plot,
                    ys_plot,
                    simulation_directory,
                    output_directory,
                    filename_current,
                    fig_size=[8,4.5],
                    fig_dpi=480,
                    imshow_kwargs=imshow_kwargs,
                    labels=plot_labels,
                    verbose=False)

"""