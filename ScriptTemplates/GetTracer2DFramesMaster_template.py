### Add Python folder to sys.path ###
import sys
sys.path.append("${ETMDirectory}")

import os
import h5py
import numpy as np
from EinsteinToolkitModules.Modules2D.Interp2D import Interp2DScalar
from EinsteinToolkitModules.Modules2D.Common2D import GetSmallestCoveringRL2D, GetDomainAfterTransforms
from EinsteinToolkitModules.Modules2D.Plot2D import PlotScalar2D
from EinsteinToolkitModules.Common import GetIterationArray, GetRefinementLevelArray, GetKeysForIteration
from EinsteinToolkitModules.Common import GetLengthFactor, GetTimeFactor, GetMassFactor, LogisticFunction

simulation_directory = "${simulationDirectory}"
input_directory = "Master/HDF5/"
file_name_rho = "rho.xy.h5"
file_name_tracer = "tracer[0].xy.h5"

output_directory = "Master/Videos/Tracer/Frames/"
file_name_o = "tracer.xy"

bounds_output = np.array([0.0,72.0,-72.0,72.0]) ##xmin, xmax, ymin, ymax
samples_output = np.array([1025,2049]) ##num_x, num_y

tracer_min = 0.0
tracer_max = 1.0
                         
transforms = ["rotate_right_half"]

if not(os.path.isdir(simulation_directory + output_directory)):
    print(simulation_directory + output_directory + " not found, creating")
    os.makedirs(simulation_directory + output_directory)

file_input_rho = h5py.File(simulation_directory + input_directory + file_name_rho,'r')
file_input_tracer = h5py.File(simulation_directory + input_directory + file_name_tracer,'r')

list_keys_rho = list(file_input_rho.keys())
list_keys_tracer = list(file_input_tracer.keys())

list_iterations_rho = GetIterationArray(list_keys_rho)
list_iterations_tracer = GetIterationArray(list_keys_tracer)

xs_output = np.linspace(bounds_output[0],bounds_output[1],num=samples_output[0])
ys_output = np.linspace(bounds_output[2],bounds_output[3],num=samples_output[1])

frame_offset = 0
#list_iterations = [0]
#list_iterations = [list_iterations[1],list_iterations[-1]]
#list_iterations = list_iterations[frame_offset:]

L_cm = GetLengthFactor("cm")
L_km = GetLengthFactor("km")
T_ms = GetTimeFactor("ms")
T_us = GetTimeFactor("us")
M_ = GetMassFactor("cgs")
Rho_cgs = M_/(L_cm**3)

Rho_mask_value = 1.0e-10*Rho_cgs
Rho_mask_k = 10.0

xs_plot, ys_plot = GetDomainAfterTransforms(xs_output,ys_output,transforms)

imshow_kwargs = {"origin":"lower",
                 "vmin":tracer_min,
                 "vmax":tracer_max,
                 "interpolation":"bessel",
                 "cmap":"inferno"}

plot_labels = {"Title":r"Tracer",
               "xAxis":r"$$x \left[ \mathrm{km} \right]$$",
               "yAxis":r"$$y \left[ \mathrm{km} \right]$$",
               "Colourbar":r"$$\mathrm{Tracer}$$"}

for it_idx in range(len(list_iterations_tracer)):
    iteration_current = list_iterations_tracer[it_idx]
    list_keys_iteration_tracer = GetKeysForIteration(iteration_current,list_keys_tracer)
    list_keys_iteration_rho = GetKeysForIteration(iteration_current,list_keys_rho)
    time_current = file_input_tracer[list_keys_iteration_tracer[0]].attrs['time']
    plot_labels["Time"] = r"$$t = " + r"{:.3f}".format(time_current*T_ms) + r" \mathrm{ms}$$"
    filename_current = file_name_o + ".fig" + str(it_idx+frame_offset).zfill(4)
    print("Creating " + filename_current + " for iteration " + str(iteration_current) + ", time = " + str(time_current))
    list_rLs_current = GetRefinementLevelArray(list_keys_iteration_tracer)
    rL_start = GetSmallestCoveringRL2D(list_rLs_current,list_keys_iteration_tracer,xs_output,ys_output,file_input_tracer)
    interp_rLs = list_rLs_current[np.nonzero(list_rLs_current == rL_start)[0][0]:]
    
    interped_data_tracer = Interp2DScalar(interp_rLs,
                                          list_keys_iteration_tracer,
                                          xs_output,
                                          ys_output,
                                          file_input_tracer,
                                          do_clip = True, 
                                          clip_bounds = np.array([tracer_min,tracer_max]), 
                                          k_spline = 1,
                                          transforms = transforms)
    
    interped_data_rho = Interp2DScalar(interp_rLs,
                                       list_keys_iteration_rho,
                                       xs_output,
                                       ys_output,
                                       file_input_rho,
                                       do_clip = True, 
                                       clip_bounds = np.array([1e-11,1]), 
                                       k_spline = 1,
                                       transforms = transforms) * Rho_cgs
                                       
    masked_tracer = LogisticFunction(np.log10(interped_data_rho),Rho_mask_k,L=interped_data_tracer,x0=np.log10(Rho_mask_value))
                                   
    PlotScalar2D(masked_tracer,
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