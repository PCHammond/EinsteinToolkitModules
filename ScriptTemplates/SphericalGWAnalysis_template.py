### Add Python folder to sys.path ###
import sys
sys.path.append("${ETMDirectory}")

### Load modules for analysis
import os
import numpy as np
from math import ceil, log
import scipy.signal as signal
from EinsteinToolkitModules.Modules1D.GW_Library import DoFixedFrequencyIntegration, LoadHarmonic, ScaleByRadius, GetRetardedTimes, GetExtrapTimes, DoExtrapStrain, GetInstantaneousFrequency, GetExtrapTimesSynced
from EinsteinToolkitModules.Modules1D.SphericalHarmonics import GetSpinWeightedSphericalYDD
from EinsteinToolkitModules.Modules1D.Rho_MaxTracking import GetMaxRhoLocs, GetFreqInstFromCoords, GetRadiiFromCoords
from EinsteinToolkitModules.Modules1D.MollweideNearestNeighbour import GetNearestNeighbourArrays
from EinsteinToolkitModules.Common import GetLengthFactor, GetTimeFactor
from scipy.interpolate import SmoothSphereBivariateSpline
import matplotlib.pyplot as plt

### Set number of samples to use for analysis
# 
interpolation_samples = 8192

# Sync strain output to time?
useTimeSync = True
limitExtrapSamples = 8192

# Set number of extrapolation samples
if not(useTimeSync):
    extrapolation_samples = 1024
else:
    syncTime = 0.0
    deltaTime = 28.8

### Options for analysis
# Get initial orbital frequency
initial_separation = ${InitialSeparation}
initial_mass = ${InitialMass}
initial_orbital_angular_frequency = (initial_mass * (initial_separation**(-3)))**(0.5)

# Approximate initial GW frequency
initial_GW_angular_frequency = 2.0*initial_orbital_angular_frequency

# Set GW lower bound to half approximate initial GW frequency
omega_cutoff = 0.5*initial_GW_angular_frequency

# Window type to use and options
window_type = "planck"
planck_epsilon = 0.1

# Ignore Weyl 4 measured inside this radius
min_radius_for_extrap = 275.0

# Load spherical harmonics up to what l
max_harmonic = 8

# Generate list of harmonics to be loaded
harmonics_to_load = []
for l in range(2,max_harmonic+1):
    for m in range(-l,l+1):
        harmonics_to_load.append([l,m])

# Number of geodesic samples to use
sample_points = 20*10**2

# Number of lat-long samples to use
theta_samples = 360
phi_samples = 720

# Radii to be loaded
radii_to_load = [#"45.00",
                 #"70.00",
                 #"100.00",
                 #"125.00",
                 #"150.00",
                 #"200.00",
                 #"250.00",
                 #"300.00",
                 #"350.00",
                 #"400.00",
                 #"500.00",
                 #"600.00",
                 "700.00"]#,
                 #"800.00",
                 #"900.00",
                 #"1000.00"]

### Directory setup
simulation_directory = "${simulationDirectory}"
input_directory = "Master/ASC/"
output_directory = "Master/Videos/SphericalGWs/Frames/"
file_name_output = "GW.ll"
fig_dpi = 480

### Verify Directory Structure ###
if not(os.path.isdir(simulation_directory)):
    print(simulation_directory + " not found, stopping")
    raise ValueError

if not(os.path.isdir(simulation_directory + input_directory)):
    print(simulation_directory + input_directory + " not found, stopping")
    raise ValueError

if not(os.path.isdir(simulation_directory + output_directory)):
    print(simulation_directory + output_directory + " not found, creating")
    os.makedirs(simulation_directory + output_directory)

### Begin analysis
print("Building nearest neighbour array")

# Get geodesic and lat-long grids, and nearest geodesic sample for each point on lat-long
geodesicSamplesCoordsThetaPhi, NearestNeighbours, Lon_edges, Lat_edges = GetNearestNeighbourArrays(sample_points,theta_samples,phi_samples)

# Number of geodesic samples
samples_used = len(geodesicSamplesCoordsThetaPhi)

# Get spherical hermonic coefficients at each geodesic sample
sYlmCoeffs = np.zeros((len(geodesicSamplesCoordsThetaPhi),len(harmonics_to_load)),dtype=complex)
print("Calculating spherical harmonic coefficients")
for sample_idx in range(len(geodesicSamplesCoordsThetaPhi)):
    for harm_idx in range(len(harmonics_to_load)):
        sYlmCoeffs[sample_idx,harm_idx] = GetSpinWeightedSphericalYDD(-2,harmonics_to_load[harm_idx][0],harmonics_to_load[harm_idx][1],geodesicSamplesCoordsThetaPhi[sample_idx,0],geodesicSamplesCoordsThetaPhi[sample_idx,1])

# Convert list of radius strings to array of float
radii = np.asarray(radii_to_load, dtype=float)

# Initialise array to hold Weyl 4 data
Weyl_4list = np.zeros(len(harmonics_to_load),dtype=object)

# Load harmonic data into list and get time domain
print("Loading Weyl4 data")
for i in range(len(harmonics_to_load)):
    times_raw, Weyl_4list[i] = LoadHarmonic(simulation_directory,input_directory,harmonics_to_load[i],radii_to_load,get_time_count = True)

# Initialise array to hold raw Weyl 4 data
Weyl4_samples = np.zeros((samples_used,)+Weyl_4list[0].shape)

# Print sample counts to IO
data_shape = Weyl4_samples.shape
print("Geodesic samples: " + str(data_shape[0]))
print("Radius samples: " + str(data_shape[1]))
print("Time samples: " + str(data_shape[2]))
print("Latitude samples: " + str(theta_samples))
print("Longitude samples: " + str(phi_samples))

print("Beginning GW analysis")

# Setup for extrapolation if syncing not used
if not(useTimeSync):
    extrap_times = np.zeros((data_shape[0],extrapolation_samples))
    extrap_reals = np.zeros((data_shape[0],extrapolation_samples))
    extrap_imags = np.zeros((data_shape[0],extrapolation_samples))

### Loop over geodesic samples
for gS_idx in range(data_shape[0]):
    print("Calculating GWs for sample point " + str(gS_idx+1) + "/" + str(data_shape[0]))

    # Initialise array to hold raw Weyl 4 data
    Weyl4_raw = np.zeros(Weyl_4list[0].shape)

    # Convert from spherical harmonics to raw data at sample location
    for i in range(len(harmonics_to_load)):
        sYlm = sYlmCoeffs[gS_idx,i]
        Weyl4_raw[:,:,0] += np.real(sYlm*(Weyl_4list[i][:,:,0]+1.0j*Weyl_4list[i][:,:,1]))
        Weyl4_raw[:,:,1] += np.imag(sYlm*(Weyl_4list[i][:,:,0]+1.0j*Weyl_4list[i][:,:,1]))
    
    # Perform FFI on Weyl 4 data
    times_interp, strain_data_interp = DoFixedFrequencyIntegration(Weyl4_raw,times_raw,interpolation_samples,omega_cutoff,window_type,window_kwargs={"epsilon":planck_epsilon})

    # Scale strain to get strain*radius
    strain_scaled_interp = ScaleByRadius(strain_data_interp,radii)
    
    # Convert time domain to retarded time
    times_retarded = GetRetardedTimes(times_interp,radii)
    
    # Set up variables for extrapolation
    radii_extrap = radii[radii>=min_radius_for_extrap]
    radii_extrap_str = np.asarray(radii_to_load)[radii>=min_radius_for_extrap]
    times_retarded_for_extrap = times_retarded[radii>=min_radius_for_extrap]
    strain_scaled_for_extrap = strain_scaled_interp[radii>=min_radius_for_extrap]
    
    # Determine extrapolation times
    if not(useTimeSync):
        times_extrap = GetExtrapTimes(times_retarded_for_extrap,extrapolation_samples,1.0-2.0*planck_epsilon)
    else:
        times_extrap = GetExtrapTimesSynced(times_retarded_for_extrap,1.0-2.0*planck_epsilon,syncTime,deltaTime)
        if gS_idx==0:
            extrapolation_samples = len(times_extrap)
            print("Extrapolation Samples set to: " + str(extrapolation_samples))
            if limitExtrapSamples:
                assert extrapolation_samples <= limitExtrapSamples
            extrap_times = np.zeros((data_shape[0],extrapolation_samples))
            extrap_reals = np.zeros((data_shape[0],extrapolation_samples))
            extrap_imags = np.zeros((data_shape[0],extrapolation_samples))
    
    # Perform extrapolation
    cplx_extrap = DoExtrapStrain(strain_scaled_for_extrap,times_retarded_for_extrap,times_extrap,radii_extrap,verbose=False)
    
    # Save data for this samples to gloobal array
    np.copyto(extrap_times[gS_idx],times_extrap)
    np.copyto(extrap_reals[gS_idx],np.real(cplx_extrap))
    np.copyto(extrap_imags[gS_idx],np.imag(cplx_extrap))

# Unit conversion
T_ms = GetTimeFactor("ms")

# Get bounds for data
real_max = np.max(extrap_reals)
real_min = np.min(extrap_reals)
imag_max = np.max(extrap_imags)
imag_min = np.min(extrap_imags)
plot_max = np.max(np.abs(np.array([real_min,real_max,imag_min,imag_max])))
cplx_plot_max = np.max(np.abs(extrap_reals + 1.0j*extrap_imags))

#splrep_thetas = np.linspace(0.0,np.pi,num = theta_samples+1)
#splrep_phis = np.linspace(0.0,2.0*np.pi,num = phi_samples+1)

#splrep_Thetas, splrep_Phis = np.meshgrid(splrep_thetas,splrep_phis,indexing="ij")
#splrep_Lats = 0.5*np.pi - splrep_Thetas
#splrep_Lons = splrep_Phis - np.pi

#weights = np.full(data_shape[0],1.0/(0.1*plot_max))
#smoothing = data_shape[0]

### Loop over time samples
for t_idx in range(extrapolation_samples):
    # Set filename for current figure
    filename_current = file_name_output + ".fig" + str(t_idx).zfill(4)

    # Determine current time
    time_current = extrap_times[0,t_idx]

    # Print info to IO
    print("Creating " + filename_current + " for time " + str("{0:.3f}".format(time_current)))
    
    # NN imterpolation of geodesic samples onto lat-long grid
    real_data = extrap_reals[NearestNeighbours,t_idx]
    imag_data = extrap_imags[NearestNeighbours,t_idx]
    
    # Convert to physical strain
    cplx_data = real_data + 1.0j*imag_data
    mag_data = np.abs(cplx_data)
    phas_data = np.arctan2(imag_data,real_data) + 2.0*np.pi*(np.arctan2(imag_data,real_data)<0.0)
    
    # Magnitude-phase diagram
    fig1 = plt.figure(1)
    fig1.set_size_inches(10,3)
    ax1a = fig1.add_subplot(1,2,1,projection="mollweide")
    ax1b = fig1.add_subplot(1,2,2,projection="mollweide")
    
    ax1a.pcolormesh(Lon_edges,Lat_edges,real_data,shading='gouraud',cmap="RdYlBu_r",vmin=-plot_max,vmax=plot_max)
    ax1a.grid(True)
    
    ax1b.pcolormesh(Lon_edges,Lat_edges,imag_data,shading='gouraud',cmap="RdYlBu_r",vmin=-plot_max,vmax=plot_max)
    ax1b.grid(True)
    
    fig1.suptitle(r"$$t = " + r"{:.3f}".format(time_current*T_ms) + r" \mathrm{ms}$$")
    ax1a.set_title(r"$$h_{+}$$")
    ax1b.set_title(r"$$h_{\times}$$")
    
    fig1.savefig(simulation_directory + output_directory + "RI/" + filename_current + ".png",dpi=fig_dpi, bbox_inches = "tight")
    
    # h_+,h_x diagram
    fig2 = plt.figure(2)
    fig2.set_size_inches(10,3)
    ax2a = fig2.add_subplot(1,2,1,projection="mollweide")
    ax2b = fig2.add_subplot(1,2,2,projection="mollweide")
    
    ax2a.pcolormesh(Lon_edges,Lat_edges,mag_data,shading='gouraud',cmap="inferno",vmin=0.0,vmax=cplx_plot_max)
    ax2a.grid(True)
    
    ax2b.pcolormesh(Lon_edges,Lat_edges,phas_data,shading='gouraud',cmap="hsv",vmin=0.0,vmax=2.0*np.pi)
    ax2b.grid(True)
    
    fig2.suptitle(r"$$t = " + r"{:.3f}".format(time_current*T_ms) + r" \mathrm{ms}$$")
    ax2a.set_title(r"$$|h_{+} + ih_{\times}|$$")
    ax2b.set_title(r"$$\mathrm{arg} \left( h_{+} + ih_{\times} \right)$$")
    
    fig2.savefig(simulation_directory + output_directory + "MP/" + filename_current + ".png",dpi=fig_dpi, bbox_inches = "tight")
    
#    fig2 = plt.figure(2)
#    fig2.set_size_inches(10,3)
#    ax2a = fig2.add_subplot(1,2,1,projection="mollweide")
#    ax2b = fig2.add_subplot(1,2,2,projection="mollweide")
#    
#    print(np.min(geodesicSamplesCoordsThetaPhi[:,0]),np.max(geodesicSamplesCoordsThetaPhi[:,0]))
#    print(np.min(geodesicSamplesCoordsThetaPhi[:,1] + 2.0*np.pi*(geodesicSamplesCoordsThetaPhi[:,1]<0.0)),
#          np.max(geodesicSamplesCoordsThetaPhi[:,1] + 2.0*np.pi*(geodesicSamplesCoordsThetaPhi[:,1]<0.0)))
#    print(geodesicSamplesCoordsThetaPhi[:,0].shape,
#          (geodesicSamplesCoordsThetaPhi[:,1] + 2.0*np.pi*(geodesicSamplesCoordsThetaPhi[:,1]<0.0)).shape,
#          extrap_reals[:,t_idx].shape)
#    
#    real_splrep = SmoothSphereBivariateSpline(geodesicSamplesCoordsThetaPhi[:,0],
#                                              geodesicSamplesCoordsThetaPhi[:,1] + 2.0*np.pi*(geodesicSamplesCoordsThetaPhi[:,1]<0.0),
#                                              extrap_reals[:,t_idx],
#                                              w=weights,s=smoothing)
#    
#    imag_splrep = SmoothSphereBivariateSpline(geodesicSamplesCoordsThetaPhi[:,0],
#                                              geodesicSamplesCoordsThetaPhi[:,1] + 2.0*np.pi*(geodesicSamplesCoordsThetaPhi[:,1]<0.0),
#                                              extrap_imags[:,t_idx],
#                                              w=weights,s=smoothing)
#    
#    real_splrep_eval = np.zeros(Lon_edges.shape)
#    imag_splrep_eval = np.zeros(Lon_edges.shape)
#    
#    real_splrep_eval = real_splrep.__call__(splrep_thetas,splrep_phis,grid=True)
#    imag_splrep_eval = imag_splrep.__call__(splrep_thetas,splrep_phis,grid=True)
#    
#    print(splrep_Lons.shape,splrep_Lats.shape,real_splrep_eval.shape)
#    
#    ax2a.pcolormesh(splrep_Lons,splrep_Lats,real_splrep_eval,shading='gouraud',cmap="RdYlBu_r",vmin=-plot_max,vmax=plot_max)
#    ax2a.grid(True)
#    
#    ax2b.pcolormesh(splrep_Lons,splrep_Lats,imag_splrep_eval,shading='gouraud',cmap="RdYlBu_r",vmin=-plot_max,vmax=plot_max)
#    ax2b.grid(True)
#    
#    fig2.suptitle(r"$$t = " + r"{:.3f}".format(time_current*T_ms) + r" \mathrm{ms}$$")
#    ax2a.set_title(r"$$h_{+}$$")
#    ax2b.set_title(r"$$h_{\times}$$")
#    
#    fig2.savefig(simulation_directory + output_directory + "SI/" + filename_current + ".png",dpi=fig_dpi, bbox_inches = "tight")
#    
#    del real_splrep
#    del imag_splrep
    
    plt.close("all")