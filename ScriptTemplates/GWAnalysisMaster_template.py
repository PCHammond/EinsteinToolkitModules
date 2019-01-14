### Add Python folder to sys.path ###
import sys
sys.path.append("${ETMDirectory}")

import os
import numpy as np
from math import ceil, log
import scipy.signal as signal
from EinsteinToolkitModules.Modules1D.GW_Library import DoFixedFrequencyIntegration, LoadHarmonic, ScaleByRadius, GetRetardedTimes, GetExtrapTimes, GetExtrapTimesSynced, DoExtrapStrain, GetInstantaneousFrequency, Weyl4Extrapolation, ExtractSignalFromWindow
from EinsteinToolkitModules.Modules1D.SphericalHarmonics import GetSpinWeightedSphericalYDD
from EinsteinToolkitModules.Modules1D.Rho_MaxTracking import GetMaxRhoLocs, GetFreqInstFromCoords, GetRadiiFromCoords
from EinsteinToolkitModules.Common import GetLengthFactor, GetTimeFactor                                                  

interpolation_samples = 8192

useTimeSync = True

if not(useTimeSync):
    extrapolation_samples = 16384
else:
    syncTime = 0.0
    deltaTime = 3.6

initial_separation = ${InitialSeparation}
initial_mass = ${InitialMass}
initial_orbital_angular_frequency = (initial_mass * (initial_separation**(-3)))**(0.5)
initial_GW_angular_frequency = 2.0*initial_orbital_angular_frequency
omega_cutoff = 0.5*initial_GW_angular_frequency
window_type = "planck"
planck_epsilon = 0.1
min_radius_for_extrap = 275.0
poly_order = 0

observation_direction = np.array([0.0,-0.5*np.pi])

harmonics_to_load = [[2,2],
                     [3,2],
                     [4,2],
                     [5,2],
                     [6,2],
                     [7,2],
                     [8,2]]

radii_to_load = ["45.00",
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

radii = np.asarray(radii_to_load, dtype=float)
Weyl_4list = np.zeros(len(harmonics_to_load),dtype=object)

for i in range(len(harmonics_to_load)):
    times_raw, Weyl_4list[i] = LoadHarmonic(simulation_directory,input_directory,harmonics_to_load[i],radii_to_load,get_time_count = True)

Weyl4_raw = np.zeros(Weyl_4list[0].shape)
for i in range(len(harmonics_to_load)):
    sYlm = GetSpinWeightedSphericalYDD(-2,harmonics_to_load[i][0],harmonics_to_load[i][1],observation_direction[0],observation_direction[1])
    print(sYlm,Weyl_4list[i].shape)
    Weyl4_raw[:,:,0] += np.real(sYlm*(Weyl_4list[i][:,:,0]+1.0j*Weyl_4list[i][:,:,1]))
    Weyl4_raw[:,:,1] += np.imag(sYlm*(Weyl_4list[i][:,:,0]+1.0j*Weyl_4list[i][:,:,1]))

print(times_raw.shape)

times_interp, strain_data_interp = DoFixedFrequencyIntegration(Weyl4_raw,times_raw,interpolation_samples,omega_cutoff,window_type,window_kwargs={"epsilon":planck_epsilon})

strain_scaled_interp = ScaleByRadius(strain_data_interp,radii)

times_retarded = GetRetardedTimes(times_interp,radii)

print("Extrapolating")

radii_extrap = radii[radii>=min_radius_for_extrap]
radii_extrap_str = np.asarray(radii_to_load)[radii>=min_radius_for_extrap]
times_retarded_for_extrap = times_retarded[radii>=min_radius_for_extrap]
strain_scaled_for_extrap = strain_scaled_interp[radii>=min_radius_for_extrap]

if not(useTimeSync):
    times_extrap = GetExtrapTimes(times_retarded_for_extrap,extrapolation_samples,1.0-2.0*planck_epsilon)
else:
    times_extrap = GetExtrapTimesSynced(times_retarded_for_extrap,1.0-2.0*planck_epsilon,syncTime,deltaTime)
    extrapolation_samples = len(times_extrap)

#times_extrap = GetExtrapTimes(times_retarded_for_extrap,extrapolation_samples,1.0-2.0*planck_epsilon)

cplx_extrap = DoExtrapStrain(strain_scaled_for_extrap,times_retarded_for_extrap,times_extrap,radii_extrap,extrap_type="poly",poly_order=3,verbose=False)

real_extrap = np.real(cplx_extrap)
imag_extrap = np.imag(cplx_extrap)
mag_extrap = np.abs(cplx_extrap)

frequency_inst = GetInstantaneousFrequency(cplx_extrap,times_extrap)

rho_max_times, rho_max_coords = GetMaxRhoLocs(simulation_directory,input_directory)
rho_max_finst = GetFreqInstFromCoords(rho_max_times, rho_max_coords)
rho_max_radii = GetRadiiFromCoords(rho_max_coords)

weyl4_extrap_times, weyl4_extrap_cplx = Weyl4Extrapolation(Weyl4_raw,
                                                           radii,
                                                           times_raw,
                                                           extrapolation_samples,
                                                           window_type,
                                                           omega_cutoff,
                                                           poly_order=3,
                                                           radii_power=-2.0,
                                                           window_kwargs={"epsilon":planck_epsilon},
                                                           verbose=False)

T_ = GetTimeFactor("cgs")
L_ = GetLengthFactor("cgs")
kmto100Mpc = 1.0/(100*3.086e+19)

print("Plotting")

import matplotlib.pyplot as plt
import matplotlib as mpl
plt.ioff()
import os
#if not('/mainfs/home/pch1g13/texlive/2018/bin/x86_64-linux' in os.environ["PATH"]):
#    os.environ["PATH"] += os.pathsep + '/mainfs/home/pch1g13/texlive/2018/bin/x86_64-linux'
#print(os.getenv("PATH"))
#plt.rc('text', usetex=False)
#plt.rc('font', family = "serif",serif = "Computer Modern Math")
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
from palettable.tableau import Tableau_20, Tableau_10
colour_list_short = Tableau_10.mpl_colors
colour_list_long = Tableau_20.mpl_colors

times_retarded_SI = times_retarded * T_
times_retarded_for_extrap_SI = times_retarded_for_extrap * T_
times_interp_SI = times_interp * T_
times_extrap_SI = times_extrap * T_
real_interp_scaled_km = np.real(strain_scaled_interp) * L_
imag_interp_scaled_km = np.imag(strain_scaled_interp) * L_
real_interp_scaled_for_extrap_km = np.real(strain_scaled_for_extrap) * L_
imag_interp_scaled_for_extrap_km = np.imag(strain_scaled_for_extrap) * L_
real_extrap_km = real_extrap * L_
imag_extrap_km = imag_extrap * L_
real_extrap_at100Mpc = real_extrap_km * kmto100Mpc
imag_extrap_at100Mpc = imag_extrap_km * kmto100Mpc
frequency_inst_SI = frequency_inst/T_
mag_extrap_at100Mpc = mag_extrap * L_ * kmto100Mpc
rho_max_times_SI = rho_max_times * T_
rho_max_finst_SI = rho_max_finst / T_
rho_max_radii_SI =  rho_max_radii * L_

fig1 = plt.figure(1)
fig1.set_size_inches(8,4.5)
ax1a = fig1.add_subplot(2,1,1)
ax1a.set_prop_cycle("color",colour_list_short)
ax1a.set_ylabel(r"$$h^{r}r (\mathrm{km})$$")
ax1b = fig1.add_subplot(2,1,2,sharex = ax1a)
ax1b.set_prop_cycle("color",colour_list_short)
ax1b.set_ylabel(r"$$h^{\infty}r (\mathrm{km})$$")
ax1b.set_xlabel(r"Time $$(\mathrm{s})$$")
real_line_start, = ax1a.plot(times_retarded_for_extrap_SI[0],real_interp_scaled_for_extrap_km[0],c="C2",ls=":")
imag_line_start, = ax1a.plot(times_retarded_for_extrap_SI[0],imag_interp_scaled_for_extrap_km[0],c="C3",ls=":")
real_line_end, = ax1a.plot(times_retarded_for_extrap_SI[-1],real_interp_scaled_for_extrap_km[-1],c="C0")
imag_line_end, = ax1a.plot(times_retarded_for_extrap_SI[-1],imag_interp_scaled_for_extrap_km[-1],c="C1")
ax1a.legend((real_line_start, 
             imag_line_start,
             real_line_end,
             imag_line_end), 
            (r"$$h^{" + radii_extrap_str[0] + r"}_{+}$$",
             r"$$h^{" + radii_extrap_str[0] + r"}_{\times}$$",
             r"$$h^{" + radii_extrap_str[-1] + r"}_{+}$$",
             r"$$h^{" + radii_extrap_str[-1] + r"}_{\times}$$"), loc="upper right")
real_line_inf, = ax1b.plot(times_extrap_SI,real_extrap_km,c="C0")
imag_line_inf, = ax1b.plot(times_extrap_SI,imag_extrap_km,c="C1")
ax1b.legend((real_line_inf, 
             imag_line_inf), 
            (r"$$h^{\infty}_{+}$$",
             r"$$h^{\infty}_{\times}$$"), loc="upper right")
plt.setp(ax1a.get_xticklabels(), visible=False)
ax1a.set_ylim([-1.0,1.0])
ax1b.set_ylim([-1.0,1.0])
plt.setp(ax1a.get_yticklabels()[0], visible=False)
plt.setp(ax1b.get_yticklabels()[-1], visible=False)
#ax1a.set_yticklabels(labs1[1::])
#ax1b.set_yticklabels(labs2[:-1:])
plt.subplots_adjust(hspace=.0)
fig1.savefig(simulation_directory + output_directory + "waveforms_extrap.png",dpi=480, bbox_inches = "tight")
plt.clf()
plt.close("all")

fig2 = plt.figure(2)
fig2.set_size_inches(8,4.5)
ax2a = fig2.add_subplot(2,1,1)
ax2b = fig2.add_subplot(2,1,2,sharex=ax2a)
ax2a.set_prop_cycle("color",colour_list_long)
ax2b.set_prop_cycle("color",colour_list_long)
for i in range(len(radii)):
    ax2a.plot(times_retarded_SI[i],real_interp_scaled_km[i])
    ax2b.plot(times_retarded_SI[i],imag_interp_scaled_km[i])
ax2a.set_ylim([-1.0,1.0])
ax2b.set_ylim([-1.0,1.0])
fig2.savefig(simulation_directory + output_directory + "waveforms_extrap_individ.png",dpi=480, bbox_inches = "tight")
plt.clf()
plt.close("all")

fig3 = plt.figure(3)
fig3.set_size_inches(8,4.5)
ax3a = fig3.add_subplot(2,1,1)
ax3b = fig3.add_subplot(2,1,2,sharex=ax2a)
ax3a.set_prop_cycle("color",colour_list_long)
ax3b.set_prop_cycle("color",colour_list_long)
for i in range(len(radii)):
    ax3a.plot(times_retarded_SI[i],np.real(strain_data_interp[i]))
    ax3b.plot(times_retarded_SI[i],np.imag(strain_data_interp[i]))
fig3.savefig(simulation_directory + output_directory + "waveforms_extrap_raw.png",dpi=480, bbox_inches = "tight")
plt.clf()
plt.close("all")

sample_frq = 1.0/(times_extrap_SI[1] - times_extrap_SI[0])

freq_res = 20.0
fft_samps = int(2**(ceil(log(sample_frq/freq_res,2.0))))
time_res_samps = int(round((times_raw[0,1]-times_raw[0,0])/(times_extrap[1]-times_extrap[0])))
if time_res_samps == 0:
    print("time_res_samps is 0, setting to 1")
    print("Desired frequency resolution will not be reached.")
    time_res_samps = 1
samps_per_seg = int(round(len(times_extrap)/8.0))
samps_overlap = samps_per_seg - time_res_samps

f,t,Sxx = signal.spectrogram(real_extrap_at100Mpc + imag_extrap_at100Mpc*1.0j,
                             fs=sample_frq,
                             window="hamming",
                             nperseg=samps_per_seg,
                             nfft=fft_samps,
                             noverlap=samps_overlap,
                             return_onesided=False)

f = np.fft.fftshift(f)
Sxx = np.fft.fftshift(Sxx,axes=0)
Sxx = Sxx[(0.0<=f)&(f<=2000.0)]
sqrtSxx = np.sqrt(Sxx)
f = f[(0.0<=f)&(f<=2000.0)]

dt = t[1]-t[0]
df = f[1]-f[0]

imshow_extent = (np.min(t) - 0.5*dt,
                 np.max(t) + 0.5*dt,
                 np.min(f) - 0.5*df,
                 np.max(f) + 0.5*df)

vmin_used = np.min(sqrtSxx)
vmax_used = np.max(sqrtSxx)

fig4 = plt.figure(4)
fig4.set_size_inches(8,4.5)
ax4a = fig4.add_subplot(1,1,1)

ax4a.set_xlabel(r"Time (s)")
ax4a.set_ylabel(r"Frequency $$(\mathrm{Hz})$$")
ax4a.set_title(r"$$\sqrt{\mathrm{PSD}}$$ at $$100\mathrm{Mpc}$$ $$(\mathrm{Hz}^{-\frac{1}{2}})$$")
im = ax4a.imshow(sqrtSxx, extent=imshow_extent, vmin=vmin_used, vmax=vmax_used, aspect="auto", origin="lower", interpolation="none")
clb = fig4.colorbar(im)
clb.set_label(r"$$\sqrt{\mathrm{PSD}}$$ $$(\mathrm{Hz}^{-\frac{1}{2}})$$")
fig4.savefig(simulation_directory + output_directory + "spectrogram_extrap.png",dpi=480, bbox_inches = "tight")
plt.clf()
plt.close("all")


fig6 = plt.figure(6)
fig6.set_size_inches(8,4.5)
ax6a = fig6.add_subplot(2,1,1)
ax6a.set_prop_cycle("color",colour_list_short)
ax6a.set_title(r"$$h^{\infty}_{+,\times}$$ Extrapolated to $$100\mathrm{Mpc}$$")
ax6b = fig6.add_subplot(2,1,2,sharex = ax6a)
ax6b.set_prop_cycle("color",colour_list_short)
ax6c = ax6b.twinx()
offset_a = -22
offset_b = -22
real_line_inf_100Mpc, = ax6a.plot(times_extrap_SI,real_extrap_at100Mpc/(10**offset_a),c="C0")
imag_line_inf_100Mpc, = ax6a.plot(times_extrap_SI,imag_extrap_at100Mpc/(10**offset_a),c="C1")
ax6a.legend((real_line_inf, 
             imag_line_inf), 
            (r"$$h^{\infty}_{+}$$",
             r"$$h^{\infty}_{\times}$$"), loc="upper right")
mag_line, = ax6b.plot(times_extrap_SI,np.abs(real_extrap_at100Mpc+1.0j*imag_extrap_at100Mpc)/(10**offset_b),c="C2")
freq_line, = ax6c.plot(times_extrap_SI,frequency_inst_SI,c="C3")
ax6b.legend((mag_line, 
             freq_line), 
            (r"$$|h^{\infty}_{+} + ih^{\infty}_{\times}|$$",
             r"$$f_{\mathrm{inst}} (\mathrm{Hz})$$"), loc="lower right")
plt.setp(ax6a.get_xticklabels(), visible=False)
ax6b.set_xlabel(r"Time $$(\mathrm{s})$$")
ax6a.set_ylim([-2.5,2.5])
ax6b.set_ylim([0.0,2.5])
plt.setp(ax6a.get_yticklabels()[0], visible=False)
plt.setp(ax6b.get_yticklabels()[-1], visible=False)
plt.subplots_adjust(hspace=.0)
fig6.canvas.draw_idle()
ax6a.set_ylabel(r"$$h^{\infty}_{+,\times} \times 10^{" + str(offset_a) + r"}$$")
ax6b.set_ylabel(r"$$|h^{\infty}_{+} + ih^{\infty}_{\times}| \times 10^{" + str(offset_b) + r"}$$")
ax6c.set_ylabel(r"$$f_{\mathrm{inst}} (\mathrm{Hz})$$")
ax6a.yaxis.offsetText.set_visible(False)
ax6b.yaxis.offsetText.set_visible(False)
fig6.savefig(simulation_directory + output_directory + "waveforms_extrap_100Mpc.png",dpi=480, bbox_inches = "tight")
plt.clf()
plt.close("all")

fig7 = plt.figure(7)
fig7.set_size_inches(8,4.5)
ax7a = fig7.add_subplot(1,1,1)
ax7a.set_prop_cycle("color",colour_list_short)
ax7a.set_xlabel(r"$$t (\mathrm{s})$$")
ax7b = ax7a.twinx()
ax7b.set_prop_cycle("color",colour_list_short)
mag_line, = ax7a.plot(times_extrap_SI,mag_extrap_at100Mpc,c="C0")
ax7a.set_ylabel(r"$$|h^{100\mathrm{Mpc}}_{+} + ih^{100\mathrm{Mpc}}_{\times}|$$")
arg_line, = ax7b.plot(times_extrap_SI,frequency_inst_SI,c="C1")
ax7b.set_ylabel(r"$$f_{\mathrm{inst}} (\mathrm{Hz})$$")
ax7a.legend((mag_line, 
             arg_line), 
            (r"$$|h^{100\mathrm{Mpc}}_{+,\times}|$$",
             r"$$f_{\mathrm{inst}}$$"), loc="best")
fig7.savefig(simulation_directory + output_directory + "mag_phaseder_extrap_100Mpc.png",dpi=480, bbox_inches = "tight")
plt.clf()
plt.close("all")

fig9 = plt.figure(9)
ax9a = fig9.add_subplot(1,1,1)
ax9a.set_prop_cycle("color",colour_list_short)
ax9a.plot(rho_max_times_SI,rho_max_radii_SI)
ax9a.set_ylabel(r"$$r (\mathrm{km})$$")
ax9a.set_xlabel(r"$$t (\mathrm{s})$$")
fig9.savefig(simulation_directory + output_directory + "radius.png",dpi=480, bbox_inches = "tight")
plt.clf()
plt.close("all")

fig10 = plt.figure(10)
ax10a = fig10.add_subplot(1,1,1)
ax10a.set_prop_cycle("color",colour_list_short)
ax10b = ax10a.twinx()
ax10b.set_prop_cycle("color",colour_list_short)
rholine, = ax10a.plot(rho_max_times_SI,rho_max_finst_SI,c="C0")
ax10a.set_ylabel(r"$$f^{\rho_{\mathrm{max}}}_{\mathrm{inst}} (\mathrm{Hz})$$")
GWline, = ax10b.plot(times_extrap_SI,frequency_inst_SI,c="C1")
ax10b.set_ylabel(r"$$f^{\mathrm{GW}}_{\mathrm{inst}} (\mathrm{Hz})$$")
ax10a.set_xlabel(r"$$t (\mathrm{s})$$")
fmin = 80.0
fmax = 500.0

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

weyl4_extrap_times_window, weyl4_extrap_cplx_window = ExtractSignalFromWindow(weyl4_extrap_times,weyl4_extrap_cplx,0.1,0.9)

weyl4_extrap_times_SI = weyl4_extrap_times_window*T_
weyl4_extrap_cplx_SI = weyl4_extrap_cplx_window * L_ * kmto100Mpc
weyl4_extrap_real_SI = np.real(weyl4_extrap_cplx_window)
weyl4_extrap_imag_SI = np.imag(weyl4_extrap_cplx_window)
weyl4_extrap_mag_SI = np.abs(weyl4_extrap_cplx_window)
weyl4_frequency_inst_SI = GetInstantaneousFrequency(weyl4_extrap_cplx_window,weyl4_extrap_times_window)/T_

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
real_line_inf_100Mpc, = ax11a.plot(weyl4_extrap_times_SI,weyl4_extrap_real_SI/(10**offset_a),c="C0")
imag_line_inf_100Mpc, = ax11a.plot(weyl4_extrap_times_SI,weyl4_extrap_imag_SI/(10**offset_a),c="C1")
ax11a.legend((real_line_inf, 
             imag_line_inf), 
            (r"$$h^{\infty}_{+}$$",
             r"$$h^{\infty}_{\times}$$"), loc="upper right")
mag_line, = ax11b.plot(weyl4_extrap_times_SI,weyl4_extrap_mag_SI/(10**offset_b),c="C2")
freq_line, = ax11c.plot(weyl4_extrap_times_SI,weyl4_frequency_inst_SI,c="C3")
ax11b.legend((mag_line, 
             freq_line), 
            (r"$$|h^{\infty}_{+} + ih^{\infty}_{\times}|$$",
             r"$$f_{\mathrm{inst}} (\mathrm{Hz})$$"), loc="lower right")
plt.setp(ax11a.get_xticklabels(), visible=False)
ax11b.set_xlabel(r"Time $$(\mathrm{s})$$")
ax11a.set_ylim([-2.5,2.5])
ax11b.set_ylim([0.0,2.5])
plt.setp(ax11a.get_yticklabels()[0], visible=False)
plt.setp(ax11b.get_yticklabels()[-1], visible=False)
plt.subplots_adjust(hspace=.0)
fig11.canvas.draw_idle()
ax11a.set_ylabel(r"$$h^{\infty}_{+,\times} \times 10^{" + str(offset_a) + r"}$$")
ax11b.set_ylabel(r"$$|h^{\infty}_{+} + ih^{\infty}_{\times}| \times 10^{" + str(offset_b) + r"}$$")
ax11c.set_ylabel(r"$$f_{\mathrm{inst}} (\mathrm{Hz})$$")
ax11a.yaxis.offsetText.set_visible(False)
ax11b.yaxis.offsetText.set_visible(False)
fig11.savefig(simulation_directory + output_directory + "waveforms_extrap_100Mpc_weyl4.png",dpi=480, bbox_inches = "tight")
plt.clf()
plt.close("all")