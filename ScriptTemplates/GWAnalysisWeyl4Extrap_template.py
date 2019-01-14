### Add Python folder to sys.path ###
import sys
sys.path.append("${ETMDirectory}")

### Load modules for analysis
import os
import numpy as np
from math import ceil, log
import scipy.signal as signal
from EinsteinToolkitModules.Modules1D.GW_Library import DoFixedFrequencyIntegration, LoadHarmonic, ScaleByRadius, GetRetardedTimes, GetExtrapTimes, GetExtrapTimesSynced, DoExtrapStrain, GetInstantaneousFrequency, Weyl4Extrapolation, ExtractSignalFromWindow, GetTimeToMerger
from EinsteinToolkitModules.Modules1D.SphericalHarmonics import GetSpinWeightedSphericalYDD
from EinsteinToolkitModules.Modules1D.Rho_MaxTracking import GetMaxRhoLocs, GetFreqInstFromCoords, GetRadiiFromCoords
from EinsteinToolkitModules.Common import GetLengthFactor, GetTimeFactor

### Set number of samples to use for analysis
# 
interpolation_samples = 8192

# Sync strain output to time?
useTimeSync = False

# Set number of extrapolation samples
if not(useTimeSync):
    extrapolation_samples = 16384*4
else:
    syncTime = 0.0
    deltaTime = 3.6

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

# Order of polynomial for extrapolation
poly_order = 0

# Location of observer in theta, phi
observation_direction = np.array([0.0,-0.5*np.pi])

# Harmonics to be loaded
harmonics_to_load = [[2,2],
                     [3,2],
                     [4,2],
                     [5,2],
                     [6,2],
                     [7,2],
                     [8,2]]

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
output_directory = "Master/Images/GWs/"

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
# Convert list of radius strings to array of float
radii = np.asarray(radii_to_load, dtype=float)

# Initialise array to hold Weyl 4 data
Weyl_4list = np.zeros(len(harmonics_to_load),dtype=object)

# Load harmonic data into list and get time domain
for i in range(len(harmonics_to_load)):
    times_raw, Weyl_4list[i] = LoadHarmonic(simulation_directory,input_directory,harmonics_to_load[i],radii_to_load,get_time_count = True)

# Convert from spherical harmonics to raw data at observer location
Weyl4_raw = np.zeros(Weyl_4list[0].shape)
for i in range(len(harmonics_to_load)):
    sYlm = GetSpinWeightedSphericalYDD(-2,harmonics_to_load[i][0],harmonics_to_load[i][1],observation_direction[0],observation_direction[1])
    Weyl4_raw[:,:,0] += np.real(sYlm*(Weyl_4list[i][:,:,0]+1.0j*Weyl_4list[i][:,:,1]))
    Weyl4_raw[:,:,1] += np.imag(sYlm*(Weyl_4list[i][:,:,0]+1.0j*Weyl_4list[i][:,:,1]))

print("Extrapolating")

# Get radii to be used for extrapolation
radii_extrap = radii[radii>=min_radius_for_extrap]
radii_extrap_str = np.asarray(radii_to_load)[radii>=min_radius_for_extrap]

# Determine motion of location of maximum density
rho_max_times, rho_max_coords = GetMaxRhoLocs(simulation_directory,input_directory)
rho_max_finst = GetFreqInstFromCoords(rho_max_times, rho_max_coords)
rho_max_radii = GetRadiiFromCoords(rho_max_coords)

# Calculate time remaining to merger from mamximum density data
times_to_merger = GetTimeToMerger(2.0*rho_max_radii, 1.2, 1.2)

# Extrapolate and integrate Weyl 4 to strain*radius 
weyl4_extrap_times, weyl4_extrap_cplx = Weyl4Extrapolation(Weyl4_raw,
                                                           radii,
                                                           times_raw,
                                                           extrapolation_samples,
                                                           window_type,
                                                           omega_cutoff,
                                                           poly_order=poly_order,
                                                           radii_power=-2.0,
                                                           window_kwargs={"epsilon":planck_epsilon},
                                                           verbose=False)

# Unit conversion
T_ = GetTimeFactor("cgs")
L_ = GetLengthFactor("cgs")
kmto100Mpc = 1.0/(100*3.086e+19)

print("Omega_cutoff = " + str((omega_cutoff/(2.0*np.pi))/T_) + " Hz")

print("Plotting")

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()
import os
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
from palettable.tableau import Tableau_20, Tableau_10
colour_list_short = Tableau_10.mpl_colors
colour_list_long = Tableau_20.mpl_colors

# Get signal untouched by window function
weyl4_extrap_times_window, weyl4_extrap_cplx_window = ExtractSignalFromWindow(weyl4_extrap_times,weyl4_extrap_cplx,0.1,0.9)

# Convert to cgs/km units
weyl4_extrap_times_SI = weyl4_extrap_times_window*T_
weyl4_extrap_cplx_SI = weyl4_extrap_cplx_window * L_ * kmto100Mpc
weyl4_extrap_real_SI = np.real(weyl4_extrap_cplx_SI)
weyl4_extrap_imag_SI = np.imag(weyl4_extrap_cplx_SI)
weyl4_extrap_mag_SI = np.abs(weyl4_extrap_cplx_SI)
weyl4_frequency_inst_SI = GetInstantaneousFrequency(weyl4_extrap_cplx_SI,weyl4_extrap_times_SI)

rho_max_times_SI = rho_max_times * T_
rho_max_finst_SI = rho_max_finst / T_
rho_max_radii_SI =  rho_max_radii * L_
times_to_merger_SI = times_to_merger * T_

# Find merger time from GW amplitude
weyl4_extrap_merger_idx = np.argmax(weyl4_extrap_mag_SI)
t_merger_SI = weyl4_extrap_times_SI[weyl4_extrap_merger_idx]
print("Merger time = " + str(t_merger_SI) + " s")

### Spectrogram analysis
sample_frq = 1.0/(weyl4_extrap_times_SI[1] - weyl4_extrap_times_SI[0])
freq_res = 20.0
fft_samps = int(2**(ceil(log(sample_frq/freq_res,2.0))))
time_res_samps = int(round((times_raw[0,1]-times_raw[0,0])/(weyl4_extrap_times_window[1]-weyl4_extrap_times_window[0])))
if time_res_samps == 0:
    print("time_res_samps is 0, setting to 1")
    print("Desired frequency resolution will not be reached.")
    time_res_samps = 1
samps_per_seg = int(round(len(weyl4_extrap_times_window)/32.0))
samps_overlap = samps_per_seg - time_res_samps

# Make spectrogram
f,t,Sxx = signal.spectrogram(weyl4_extrap_cplx_SI,
                             fs=sample_frq,
                             window="hamming",
                             nperseg=samps_per_seg,
                             nfft=fft_samps,
                             noverlap=samps_overlap,
                             return_onesided=False)

f = np.fft.fftshift(f)
Sxx = np.fft.fftshift(Sxx,axes=0)
Sxx = Sxx[(0.0<=f)&(f<=3000.0)]
sqrtSxx = np.sqrt(Sxx)
f = f[(0.0<=f)&(f<=3000.0)]

dt = t[1]-t[0]
df = f[1]-f[0]

imshow_extent = (np.min(t) - 0.5*dt,
                 np.max(t) + 0.5*dt,
                 np.min(f) - 0.5*df,
                 np.max(f) + 0.5*df)

vmin_used = np.min(sqrtSxx)
vmax_used = np.max(sqrtSxx)

fig4a = plt.figure(4)
fig4a.set_size_inches(8,4.5)
ax4a = fig4a.add_subplot(1,1,1)

ax4a.set_xlabel(r"Time (s)")
ax4a.set_ylabel(r"Frequency $$(\mathrm{Hz})$$")
ax4a.set_title(r"$$\sqrt{\mathrm{PSD}}$$ at $$100\mathrm{Mpc}$$ $$(\mathrm{Hz}^{-\frac{1}{2}})$$")
im = ax4a.imshow(sqrtSxx, extent=imshow_extent, vmin=vmin_used, vmax=vmax_used, aspect="auto", origin="lower", interpolation="none")
clb = fig4a.colorbar(im)
clb.set_label(r"$$\sqrt{\mathrm{PSD}}$$ $$(\mathrm{Hz}^{-\frac{1}{2}})$$")
fig4a.savefig(simulation_directory + output_directory + "spectrogram_extrap.png",dpi=480, bbox_inches = "tight")
plt.clf()
plt.close("all")

fig4b = plt.figure(4)
fig4b.set_size_inches(8,4.5)
ax4b = fig4b.add_subplot(1,1,1)

ax4b.set_xlabel(r"Time (s)")
ax4b.set_ylabel(r"Frequency $$(\mathrm{Hz})$$")
ax4b.set_title(r"$$\sqrt{\mathrm{PSD}}$$ at $$100\mathrm{Mpc}$$ $$(\mathrm{Hz}^{-\frac{1}{2}})$$")
im = ax4b.imshow(np.log10(sqrtSxx), extent=imshow_extent, vmin=np.log10(vmin_used), vmax=np.log10(vmax_used), aspect="auto", origin="lower", interpolation="none")
clb = fig4b.colorbar(im)
clb.set_label(r"$$\sqrt{\mathrm{PSD}}$$ $$(\mathrm{Hz}^{-\frac{1}{2}})$$")
fig4b.savefig(simulation_directory + output_directory + "spectrogram_log_extrap.png",dpi=480, bbox_inches = "tight")
plt.clf()
plt.close("all")

# Plot radial distance to maximum density and time remaining until merger
fig9 = plt.figure(9)
ax9a = fig9.add_subplot(1,1,1)
ax9b = ax9a.twinx()
ax9a.set_prop_cycle("color",colour_list_short)
ax9a.plot(rho_max_times_SI,rho_max_radii_SI,c="C0",lw=0.5)
ax9a.set_ylim([0.0,np.max(rho_max_radii_SI)])
ax9a.set_ylabel(r"$$r (\mathrm{km})$$")
ax9a.set_xlabel(r"$$t (\mathrm{s})$$")
ax9a.set_xlim([rho_max_times_SI[0],rho_max_times_SI[-1]])
ax9b.set_prop_cycle("color",colour_list_short)
ax9b.plot(rho_max_times_SI,times_to_merger_SI + rho_max_times_SI,c="C1",lw=0.5)
time_plot_limit = np.max(times_to_merger_SI + rho_max_times_SI)
ax9b.set_ylim([t_merger_SI,time_plot_limit])
ax9b.set_ylabel(r"$$ t_{\mathrm{merger}} \left[ \mathrm{s} \right] $$")
fig9.savefig(simulation_directory + output_directory + "radius.png",dpi=480, bbox_inches = "tight")
plt.clf()
plt.close("all")

# Plot instantaneous frequency of rotation of maximum density about origin, and GW instantaneous frequency
fig10 = plt.figure(10)
ax10a = fig10.add_subplot(1,1,1)
ax10a.set_prop_cycle("color",colour_list_short)
ax10b = ax10a.twinx()
ax10b.set_prop_cycle("color",colour_list_short)
rholine, = ax10a.plot(rho_max_times_SI,rho_max_finst_SI,c="C0",lw=0.5)
ax10a.set_ylabel(r"$$f^{\rho_{\mathrm{max}}}_{\mathrm{inst}} (\mathrm{Hz})$$")
GWline, = ax10b.plot(weyl4_extrap_times_SI,weyl4_frequency_inst_SI,c="C1",lw=0.5)
ax10b.set_ylabel(r"$$f^{\mathrm{GW}}_{\mathrm{inst}} (\mathrm{Hz})$$")
ax10a.set_xlabel(r"$$t (\mathrm{s})$$")
fmin = 0.0
fmax = 1500.0

lablocsa = np.linspace(fmin,fmax,num=7)
lablocsb = np.linspace(2*fmin,2*fmax,num=7)
labstrsa = []
labstrsb = []
for i in range(len(lablocsa)):
    labstrsa.append(r"$$%.1f$$" % lablocsa[i])
    labstrsb.append(r"$$%.1f$$" % lablocsb[i])

ax10a.set_ylim([fmin,fmax])
ax10b.set_ylim([2*fmin,2*fmax])
ax10a.set_yticks(lablocsa)
ax10b.set_yticks(lablocsb)
ax10a.legend((rholine,
             GWline),
            (r"$$\rho_{\mathrm{max}}$$",
             r"$$\mathrm{GW}$$"), loc="upper left")
fig10.savefig(simulation_directory + output_directory + "f_inst.png",dpi=480, bbox_inches = "tight")
plt.clf()
plt.close("all")

# Plot strain of GW extrapolated out to 100Mpc
fig11 = plt.figure(11)
fig11.set_size_inches(8,4.5)
ax11a = fig11.add_subplot(2,1,1)
ax11a.set_prop_cycle("color",colour_list_short)
ax11a.set_title(r"$$h^{\infty}_{+,\times}$$ Extrapolated to $$100\mathrm{Mpc}$$")
ax11b = fig11.add_subplot(2,1,2,sharex = ax11a)
ax11b.set_prop_cycle("color",colour_list_short)
ax11c = ax11b.twinx()
offset_a = -22
offset_b = -22
real_line_inf_100Mpc, = ax11a.plot(weyl4_extrap_times_SI,weyl4_extrap_real_SI/(10**offset_a),c="C0",lw=0.5)
imag_line_inf_100Mpc, = ax11a.plot(weyl4_extrap_times_SI,weyl4_extrap_imag_SI/(10**offset_a),c="C1",lw=0.5)
ax11a.legend((real_line_inf_100Mpc,
             imag_line_inf_100Mpc),
            (r"$$h^{\infty}_{+}$$",
             r"$$h^{\infty}_{\times}$$"), loc="upper left")
mag_line, = ax11b.plot(weyl4_extrap_times_SI,weyl4_extrap_mag_SI/(10**offset_b),c="C2",lw=0.5)
freq_line, = ax11c.plot(weyl4_extrap_times_SI,weyl4_frequency_inst_SI,c="C3",lw=0.5)
ax11b.legend((mag_line,
             freq_line),
            (r"$$|h^{\infty}_{+} + ih^{\infty}_{\times}|$$",
             r"$$f_{\mathrm{inst}} (\mathrm{Hz})$$"), loc="upper left")
plt.setp(ax11a.get_xticklabels(), visible=False)
ax11b.set_xlabel(r"Time $$(\mathrm{s})$$")
ax11a.set_ylim([-2.5,2.5])
ax11b.set_ylim([0.0,2.5])
ax11c.set_ylim([0.0,4000.0])
plt.setp(ax11a.get_yticklabels()[0], visible=False)
plt.setp(ax11b.get_yticklabels()[-1], visible=False)
plt.subplots_adjust(hspace=.0)
fig11.canvas.draw_idle()
ax11a.set_ylabel(r"$$h^{\infty}_{+,\times} \times 10^{" + str(offset_a) + r"}$$")
ax11b.set_ylabel(r"$$|h^{\infty}_{+} + ih^{\infty}_{\times}| \times 10^{" + str(offset_b) + r"}$$")
ax11c.set_ylabel(r"$$f_{\mathrm{inst}} (\mathrm{Hz})$$")
ax11a.yaxis.offsetText.set_visible(False)
ax11b.yaxis.offsetText.set_visible(False)
fig11.savefig(simulation_directory + output_directory + "waveforms_extrap_100Mpc.png",dpi=480, bbox_inches = "tight")
plt.clf()
plt.close("all")

# Plot strain of GW extrapolated out to 100Mpc after merger
fig12 = plt.figure(12)
fig12.set_size_inches(8,4.5)
ax12a = fig12.add_subplot(2,1,1)
ax12a.set_prop_cycle("color",colour_list_short)
ax12a.set_title(r"$$h^{\infty}_{+,\times}$$ Extrapolated to $$100\mathrm{Mpc}$$")
ax12b = fig12.add_subplot(2,1,2,sharex = ax12a)
ax12b.set_prop_cycle("color",colour_list_short)
ax12c = ax12b.twinx()
offset_a = -22
offset_b = -22
real_line_inf_100Mpc, = ax12a.plot(weyl4_extrap_times_SI[weyl4_extrap_merger_idx:],
                                   weyl4_extrap_real_SI[weyl4_extrap_merger_idx:]/(10**offset_a),
                                   c="C0",lw=0.5)
imag_line_inf_100Mpc, = ax12a.plot(weyl4_extrap_times_SI[weyl4_extrap_merger_idx:],
                                   weyl4_extrap_imag_SI[weyl4_extrap_merger_idx:]/(10**offset_a),
                                   c="C1",lw=0.5)
ax12a.legend((real_line_inf_100Mpc,
             imag_line_inf_100Mpc),
            (r"$$h^{\infty}_{+}$$",
             r"$$h^{\infty}_{\times}$$"), loc="upper left")
mag_line, = ax12b.plot(weyl4_extrap_times_SI[weyl4_extrap_merger_idx:],
                       weyl4_extrap_mag_SI[weyl4_extrap_merger_idx:]/(10**offset_b),
                       c="C2",lw=0.5)
freq_line, = ax12c.plot(weyl4_extrap_times_SI[weyl4_extrap_merger_idx:],
                        weyl4_frequency_inst_SI[weyl4_extrap_merger_idx:],
                        c="C3",lw=0.5)
ax12b.legend((mag_line,
             freq_line),
            (r"$$|h^{\infty}_{+} + ih^{\infty}_{\times}|$$",
             r"$$f_{\mathrm{inst}} (\mathrm{Hz})$$"), loc="upper left")
plt.setp(ax12a.get_xticklabels(), visible=False)
ax12b.set_xlabel(r"Time $$(\mathrm{s})$$")
ax12a.set_ylim([-0.75,0.75])
ax12b.set_ylim([0.0,0.75])
ax12c.set_ylim([0.0,4000.0])
plt.setp(ax12a.get_yticklabels()[0], visible=False)
plt.setp(ax12b.get_yticklabels()[-1], visible=False)
plt.subplots_adjust(hspace=.0)
fig12.canvas.draw_idle()
ax12a.set_ylabel(r"$$h^{\infty}_{+,\times} \times 10^{" + str(offset_a) + r"}$$")
ax12b.set_ylabel(r"$$|h^{\infty}_{+} + ih^{\infty}_{\times}| \times 10^{" + str(offset_b) + r"}$$")
ax12c.set_ylabel(r"$$f_{\mathrm{inst}} (\mathrm{Hz})$$")
ax12a.yaxis.offsetText.set_visible(False)
ax12b.yaxis.offsetText.set_visible(False)
fig12.savefig(simulation_directory + output_directory + "waveforms_extrap_100Mpc_postMerger.png",dpi=480, bbox_inches = "tight")
plt.clf()
plt.close("all")

# Plot strain of GW extrapolated out to 100Mpc before merger
fig13 = plt.figure(13)
fig13.set_size_inches(8,4.5)
ax13a = fig13.add_subplot(2,1,1)
ax13a.set_prop_cycle("color",colour_list_short)
ax13a.set_title(r"$$h^{\infty}_{+,\times}$$ Extrapolated to $$100\mathrm{Mpc}$$")
ax13b = fig13.add_subplot(2,1,2,sharex = ax13a)
ax13b.set_prop_cycle("color",colour_list_short)
ax13c = ax13b.twinx()
offset_a = -22
offset_b = -22
real_line_inf_100Mpc, = ax13a.plot(weyl4_extrap_times_SI[:weyl4_extrap_merger_idx],
                                   weyl4_extrap_real_SI[:weyl4_extrap_merger_idx]/(10**offset_a),
                                   c="C0",lw=0.5)
imag_line_inf_100Mpc, = ax13a.plot(weyl4_extrap_times_SI[:weyl4_extrap_merger_idx],
                                   weyl4_extrap_imag_SI[:weyl4_extrap_merger_idx]/(10**offset_a),
                                   c="C1",lw=0.5)
ax13a.legend((real_line_inf_100Mpc,
             imag_line_inf_100Mpc),
            (r"$$h^{\infty}_{+}$$",
             r"$$h^{\infty}_{\times}$$"), loc="upper left")
mag_line, = ax13b.plot(weyl4_extrap_times_SI[:weyl4_extrap_merger_idx],
                       weyl4_extrap_mag_SI[:weyl4_extrap_merger_idx]/(10**offset_b),
                       c="C2",lw=0.5)
freq_line, = ax13c.plot(weyl4_extrap_times_SI[:weyl4_extrap_merger_idx],
                        weyl4_frequency_inst_SI[:weyl4_extrap_merger_idx],
                        c="C3",lw=0.5)
ax13b.legend((mag_line,
             freq_line),
            (r"$$|h^{\infty}_{+} + ih^{\infty}_{\times}|$$",
             r"$$f_{\mathrm{inst}} (\mathrm{Hz})$$"), loc="upper left")
plt.setp(ax13a.get_xticklabels(), visible=False)
ax13b.set_xlabel(r"Time $$(\mathrm{s})$$")
ax13a.set_ylim([-2.0,2.0])
ax13b.set_ylim([0.0,2.0])
ax13c.set_ylim([0.0,np.max(weyl4_frequency_inst_SI[:weyl4_extrap_merger_idx])])
plt.setp(ax13a.get_yticklabels()[0], visible=False)
plt.setp(ax13b.get_yticklabels()[-1], visible=False)
plt.subplots_adjust(hspace=.0)
fig13.canvas.draw_idle()
ax13a.set_ylabel(r"$$h^{\infty}_{+,\times} \times 10^{" + str(offset_a) + r"}$$")
ax13b.set_ylabel(r"$$|h^{\infty}_{+} + ih^{\infty}_{\times}| \times 10^{" + str(offset_b) + r"}$$")
ax13c.set_ylabel(r"$$f_{\mathrm{inst}} (\mathrm{Hz})$$")
ax13a.yaxis.offsetText.set_visible(False)
ax13b.yaxis.offsetText.set_visible(False)
fig13.savefig(simulation_directory + output_directory + "waveforms_extrap_100Mpc_preMerger.png",dpi=480, bbox_inches = "tight")
plt.clf()
plt.close("all")

#times_retarded_SI = times_retarded * T_
#times_retarded_for_extrap_SI = times_retarded_for_extrap * T_
#times_interp_SI = times_interp * T_
#times_extrap_SI = times_extrap * T_
#real_interp_scaled_km = np.real(strain_scaled_interp) * L_
#imag_interp_scaled_km = np.imag(strain_scaled_interp) * L_
#real_interp_scaled_for_extrap_km = np.real(strain_scaled_for_extrap) * L_
#imag_interp_scaled_for_extrap_km = np.imag(strain_scaled_for_extrap) * L_
#real_extrap_km = real_extrap * L_
#imag_extrap_km = imag_extrap * L_
#real_extrap_at100Mpc = real_extrap_km * kmto100Mpc
#imag_extrap_at100Mpc = imag_extrap_km * kmto100Mpc
#frequency_inst_SI = frequency_inst/T_
#mag_extrap_at100Mpc = mag_extrap * L_ * kmto100Mpc


#fig1 = plt.figure(1)
#fig1.set_size_inches(8,4.5)
#ax1a = fig1.add_subplot(2,1,1)
#ax1a.set_prop_cycle("color",colour_list_short)
#ax1a.set_ylabel(r"$$h^{r}r (\mathrm{km})$$")
#ax1b = fig1.add_subplot(2,1,2,sharex = ax1a)
#ax1b.set_prop_cycle("color",colour_list_short)
#ax1b.set_ylabel(r"$$h^{\infty}r (\mathrm{km})$$")
#ax1b.set_xlabel(r"Time $$(\mathrm{s})$$")
#real_line_start, = ax1a.plot(times_retarded_for_extrap_SI[0],real_interp_scaled_for_extrap_km[0],c="C2",ls=":")
#imag_line_start, = ax1a.plot(times_retarded_for_extrap_SI[0],imag_interp_scaled_for_extrap_km[0],c="C3",ls=":")
#real_line_end, = ax1a.plot(times_retarded_for_extrap_SI[-1],real_interp_scaled_for_extrap_km[-1],c="C0")
#imag_line_end, = ax1a.plot(times_retarded_for_extrap_SI[-1],imag_interp_scaled_for_extrap_km[-1],c="C1")
#ax1a.legend((real_line_start,
#             imag_line_start,
#             real_line_end,
#             imag_line_end),
#            (r"$$h^{" + radii_extrap_str[0] + r"}_{+}$$",
#             r"$$h^{" + radii_extrap_str[0] + r"}_{\times}$$",
#             r"$$h^{" + radii_extrap_str[-1] + r"}_{+}$$",
#             r"$$h^{" + radii_extrap_str[-1] + r"}_{\times}$$"), loc="upper right")
#real_line_inf, = ax1b.plot(times_extrap_SI,real_extrap_km,c="C0")
#imag_line_inf, = ax1b.plot(times_extrap_SI,imag_extrap_km,c="C1")
#ax1b.legend((real_line_inf,
#             imag_line_inf),
#            (r"$$h^{\infty}_{+}$$",
#             r"$$h^{\infty}_{\times}$$"), loc="upper right")
#plt.setp(ax1a.get_xticklabels(), visible=False)
#ax1a.set_ylim([-1.0,1.0])
#ax1b.set_ylim([-1.0,1.0])
#plt.setp(ax1a.get_yticklabels()[0], visible=False)
#plt.setp(ax1b.get_yticklabels()[-1], visible=False)
##ax1a.set_yticklabels(labs1[1::])
##ax1b.set_yticklabels(labs2[:-1:])
#plt.subplots_adjust(hspace=.0)
#fig1.savefig(simulation_directory + output_directory + "waveforms_extrap.png",dpi=480, bbox_inches = "tight")
#plt.clf()
#plt.close("all")

#fig2 = plt.figure(2)
#fig2.set_size_inches(8,4.5)
#ax2a = fig2.add_subplot(2,1,1)
#ax2b = fig2.add_subplot(2,1,2,sharex=ax2a)
#ax2a.set_prop_cycle("color",colour_list_long)
#ax2b.set_prop_cycle("color",colour_list_long)
#for i in range(len(radii)):
#    ax2a.plot(times_retarded_SI[i],real_interp_scaled_km[i])
#    ax2b.plot(times_retarded_SI[i],imag_interp_scaled_km[i])
#ax2a.set_ylim([-1.0,1.0])
#ax2b.set_ylim([-1.0,1.0])
#fig2.savefig(simulation_directory + output_directory + "waveforms_extrap_individ.png",dpi=480, bbox_inches = "tight")
#plt.clf()
#plt.close("all")

#fig3 = plt.figure(3)
#fig3.set_size_inches(8,4.5)
#ax3a = fig3.add_subplot(2,1,1)
#ax3b = fig3.add_subplot(2,1,2,sharex=ax2a)
#ax3a.set_prop_cycle("color",colour_list_long)
#ax3b.set_prop_cycle("color",colour_list_long)
#for i in range(len(radii)):
#    ax3a.plot(times_retarded_SI[i],np.real(strain_data_interp[i]))
#    ax3b.plot(times_retarded_SI[i],np.imag(strain_data_interp[i]))
#fig3.savefig(simulation_directory + output_directory + "waveforms_extrap_raw.png",dpi=480, bbox_inches = "tight")
#plt.clf()
#plt.close("all")

#fig6 = plt.figure(6)
#fig6.set_size_inches(8,4.5)
#ax6a = fig6.add_subplot(2,1,1)
#ax6a.set_prop_cycle("color",colour_list_short)
#ax6a.set_title(r"$$h^{\infty}_{+,\times}$$ Extrapolated to $$100\mathrm{Mpc}$$")
#ax6b = fig6.add_subplot(2,1,2,sharex = ax6a)
#ax6b.set_prop_cycle("color",colour_list_short)
#ax6c = ax6b.twinx()
#offset_a = -22
#offset_b = -22
#real_line_inf_100Mpc, = ax6a.plot(weyl4_extrap_times_SI,weyl4_extrap_real_SI/(10**offset_a),c="C0")
#imag_line_inf_100Mpc, = ax6a.plot(weyl4_extrap_times_SI,weyl4_extrap_imag_SI/(10**offset_a),c="C1")
#ax6a.legend((real_line_inf_100Mpc,
#             imag_line_inf_100Mpc),
#            (r"$$h^{\infty}_{+}$$",
#             r"$$h^{\infty}_{\times}$$"), loc="upper right")
#mag_line, = ax6b.plot(weyl4_extrap_times_SI,np.abs(real_extrap_at100Mpc+1.0j*imag_extrap_at100Mpc)/(10**offset_b),c="C2")
#freq_line, = ax6c.plot(weyl4_extrap_times_SI,frequency_inst_SI,c="C3")
#ax6b.legend((mag_line,
#             freq_line),
#            (r"$$|h^{\infty}_{+} + ih^{\infty}_{\times}|$$",
#             r"$$f_{\mathrm{inst}} (\mathrm{Hz})$$"), loc="lower right")
#plt.setp(ax6a.get_xticklabels(), visible=False)
#ax6b.set_xlabel(r"Time $$(\mathrm{s})$$")
#ax6a.set_ylim([-2.5,2.5])
#ax6b.set_ylim([0.0,2.5])
#plt.setp(ax6a.get_yticklabels()[0], visible=False)
#plt.setp(ax6b.get_yticklabels()[-1], visible=False)
#plt.subplots_adjust(hspace=.0)
#fig6.canvas.draw_idle()
#ax6a.set_ylabel(r"$$h^{\infty}_{+,\times} \times 10^{" + str(offset_a) + r"}$$")
#ax6b.set_ylabel(r"$$|h^{\infty}_{+} + ih^{\infty}_{\times}| \times 10^{" + str(offset_b) + r"}$$")
#ax6c.set_ylabel(r"$$f_{\mathrm{inst}} (\mathrm{Hz})$$")
#ax6a.yaxis.offsetText.set_visible(False)
#ax6b.yaxis.offsetText.set_visible(False)
#fig6.savefig(simulation_directory + output_directory + "waveforms_extrap_100Mpc.png",dpi=480, bbox_inches = "tight")
#plt.clf()
#plt.close("all")

#fig7 = plt.figure(7)
#fig7.set_size_inches(8,4.5)
#ax7a = fig7.add_subplot(1,1,1)
#ax7a.set_prop_cycle("color",colour_list_short)
#ax7a.set_xlabel(r"$$t (\mathrm{s})$$")
#ax7b = ax7a.twinx()
#ax7b.set_prop_cycle("color",colour_list_short)
#mag_line, = ax7a.plot(times_extrap_SI,mag_extrap_at100Mpc,c="C0")
#ax7a.set_ylabel(r"$$|h^{100\mathrm{Mpc}}_{+} + ih^{100\mathrm{Mpc}}_{\times}|$$")
#arg_line, = ax7b.plot(times_extrap_SI,frequency_inst_SI,c="C1")
#ax7b.set_ylabel(r"$$f_{\mathrm{inst}} (\mathrm{Hz})$$")
#ax7a.legend((mag_line,
#             arg_line),
#            (r"$$|h^{100\mathrm{Mpc}}_{+,\times}|$$",
#             r"$$f_{\mathrm{inst}}$$"), loc="best")
#fig7.savefig(simulation_directory + output_directory + "mag_phaseder_extrap_100Mpc.png",dpi=480, bbox_inches = "tight")
#plt.clf()
#plt.close("all")
