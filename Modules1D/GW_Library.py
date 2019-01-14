#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 11:30:34 2018

@author: pete
"""

import numpy as np
import scipy.signal as signal
import scipy.interpolate as interp
from math import floor, ceil

def GetTimeToMerger(separations, m1, m2):
    """
    Remaining time to merger based on quadrupole approximation.

    Args:
    separations - array(float) - array containg neutron star separations
    m1 - float - mass of larger neutron star
    m2 - float - mass of smaller neutron star

    Returns:
    array(float) - times until merger of the neutron stars at corresponding separation
    """
    return (5/256) * (separations**4.0) / ((m1 * m2) * (m1 + m2))

def Get_Omegas(fft_freq,cutoff_omega,cutoff_power = 1.0):
    """
    Omegas for Fixed Frequency Integration of Weyl4.
    
    Args:
    fft_freq - array(float) - frequencies of fourier transform
    cutoff_omega - float - cutoff omega for filter

    Kwargs:
    cutoff_power=1.0 - float - strength of filter

    Returns:
    array(float) - omegas to use for FFI

    For omega>=cutoff_omega, function returns omega, otherwise function returns omega*(cutoff_omega/omega)**cutoff_power. 
    For omega=0, function returns cutoff_omega
    """
    omegas = 2.0*np.pi*fft_freq
    output = np.zeros(omegas.shape)
    for i in range(len(omegas)):
        if np.abs(omegas[i]) >= cutoff_omega:
            output[i] = omegas[i]
        elif omegas[i] == 0.0:
            output[i] = cutoff_omega
        else:
            output[i] = omegas[i] * (cutoff_omega/omegas[i])**cutoff_power

    #output = np.zeros(omegas.shape)
    #output += (np.abs(omegas)>=cutoff_omega)*omegas
    #output += (((np.abs(omegas)<cutoff_omega) * (omegas>=0.0))*cutoff_omega)
    #output -= (((np.abs(omegas)<cutoff_omega) * (omegas<0.0))*cutoff_omega)
    return output

def Zed(n,N,e,s):
    """
    Helper function for Planck_Taper_Window
    """
    try:
        first = 1.0/(1.0 + s*(2.0*n/(N-1)-1))
    except:
        first = np.inf
    try:
        second = 1.0/(1.0-2.0*e + s*(2.0*n/(N-1)-1))
    except:
        second = np.inf
    return 2.0*e*(first + second)
    
def Planck_Taper_Window(samples,epsilon=0.1):
    """
    Planck window fuction for fourier analysis.

    Args:
    samples - int - length of window in samples

    Kwargs:
    epsilon=0.1 - float - fraction of window taken up by ramp at each end of window

    Returns:
    array(float) - Planck window of length=samples
    """
    output = np.ones(samples)
    for n in range(samples):
        if (n<epsilon*(samples-1)):
            output[n] = 1.0/(np.exp(Zed(n,samples,epsilon,1.0))+1.0)
        elif ((1-epsilon)*(samples-1)<n):
            output[n] = 1.0/(np.exp(Zed(n,samples,epsilon,-1.0))+1.0)
    return output

def LoadHarmonic(simulation_directory,input_directory,harmonic,radii_str_list,time_count = None,get_time_count = True):
    """
    Load a particular Weyl4 harmonic from file.

    Args:
    simulation_directory - str - absolute location of simulation data
    input_directory - str - relative location of data fille to load from simulation_directory
    harmonic - list(int) - spherical harmonic to load in form [l,m]
    radii_str_list - list(str) - radii at which Weyl4 was measured as strings

    Kwargs:
    time_count=None - int - number of time samples if known
    get_time_count=True - bool - read number of time samples from data file

    Returns:
    array(float)[radius,time] - real part of Weyl4 for harmonic
    array(float)[radius,time] - imaginary part of Weyl4 for harmonic
    """
    filename_base = "mp_Psi4_l" + str(harmonic[0]) + "_m" + str(harmonic[1]) + "_r"
    if get_time_count:
        testarray = np.load(simulation_directory + input_directory + filename_base + radii_str_list[0] + ".npy")
        time_count = len(testarray[:,0])
    elif not(time_count):
        raise
    data_raw = np.zeros((len(radii_str_list),time_count,3))
    for rad_idx in range(len(radii_str_list)):
        data_raw[rad_idx,:,:] = np.load(simulation_directory + input_directory + filename_base + radii_str_list[rad_idx] + ".npy")
    return data_raw[:,:,0], data_raw[:,:,1:]

def GetWindowingFunction(window_type, samples, **kwargs):
    """
    Wrapper for different window functions.

    Args:
    window_type - str - window type to use
    samples - int - length of window to generate

    Kwargs:
    **kwargs - dict - dictionary of kwargs to pass to window generation function
    
    Returns:
    array(float) - window of length=samples of chosen type

    Allowed values of window_type are:
    hann - Hann window
    tukey - Tukey window - takes optional parameter alpha
    planck - Planck window - takes optional parameter epsilon
    """
    
    if window_type=="hann": 
        window_arr = signal.windows.hann(samples,kwargs)
    elif window_type=="tukey":
        window_arr = signal.windows.tukey(samples,kwargs["alpha"])
    elif window_type=="planck":
        window_arr = Planck_Taper_Window(samples,epsilon=kwargs["epsilon"])
    else: 
        window_arr = np.ones(samples)
    return window_arr

def DoFixedFrequencyIntegration(Weyl4_data,times_data,time_samples,omega_cutoff,window_type,window_kwargs = None):
    """
    Fixed Frequency Integration of Weyl4 to obtain strain.

    Args:
    Weyl4_data - array(float)[radius,time,part] - Weyl4 data array containing both real and imaginary parts at each radius and time
    times_data - array(float) - times of Weyl4 samples
    time_samples - int - desired number of time samples
    omega_cutoff - float - cutoff angluar frequency to pass to Get_Omegas
    window_type - str - window type to use

    Kwargs:
    window_kwargs=None - dict - kwargs for window generation

    Returns:
    array(float) - interpolated times
    array(complex)[radius,time] - interpolated and integrated data
    """
    radius_samples = len(Weyl4_data)
    window_arr = GetWindowingFunction(window_type, time_samples, **window_kwargs)
    mix_corrected = np.zeros((radius_samples,time_samples),dtype=complex)
    times_interp = np.zeros((radius_samples,time_samples))
    
    for i in range(radius_samples):
        times_interp[i] = np.linspace(times_data[i,0],times_data[i,-1],num = time_samples)
        data_raw_real_splinerep = interp.interp1d(times_data[i],-1.0*Weyl4_data[i,:,0], kind="cubic")
        data_raw_imag_splinerep = interp.interp1d(times_data[i],Weyl4_data[i,:,1], kind="cubic")
        unwindowed_data = (data_raw_real_splinerep(times_interp[i]) + 1.0j*data_raw_imag_splinerep(times_interp[i]))
        interp_data = window_arr * unwindowed_data
        
        fft_interp_mix  = np.fft.fftshift(np.fft.fft(interp_data))
        fft_interp_freq = np.fft.fftshift(np.fft.fftfreq(time_samples,d=times_interp[i,1]-times_interp[i,0]))
        
        omegas = Get_Omegas(fft_interp_freq,omega_cutoff)
        
        fft_integ2 = -fft_interp_mix/(omegas**2)
        
        mix_corrected[i] = np.fft.ifft(np.fft.ifftshift(fft_integ2))
        
    return times_interp, mix_corrected

def GetRetardedTimes(times,radii):
    """
    Calculate retarded times.

    Args:
    times - array(float)[radius,time] - times in global frame
    radii - array(float)[radius] - radius for each detector

    Returns:
    array(float)[radius,time] - retared times at each radius
    """
    output = np.zeros(times.shape)
    
    for i in range(len(radii)):
        output[i] = times[i] - radii[i]
    return output

def ScaleByRadius(strain_data,radii):
    output = np.zeros(strain_data.shape,dtype=complex)
    
    for i in range(len(radii)):
        output[i] = strain_data[i] * radii[i]
        
    return output

def GetExtrapTimes(times_retarded,time_samples,window_width):
    """
    Calculate times for extrapolation that avoid data changed by windowing.

    Args:
    times_retarded - array(float)[radius,time] - retared times for each detector
    time_samples - int - desired number of time samples
    window_width - float - fractional length of window unaltered by windowing function. Window function assumed symmetrical.

    Returns:
    array(float)[time_samples] - time samples that are inside window for all radius samples
    """
    t_min = np.min(times_retarded[ 0]) + 0.5*(1.0-window_width)*(times_retarded[ 0][-1]-times_retarded[ 0][0])
    t_max = np.max(times_retarded[-1]) - 0.5*(1.0-window_width)*(times_retarded[-1][-1]-times_retarded[-1][0])
    
    times_extrap = np.linspace(t_min,t_max,num = time_samples)
    
    return times_extrap

def GetExtrapTimesSynced(times_retarded,window_width,t0,t_delta):
    """
    Calculate times for extrapolation that avois data changed by windowing and are synced to t = t0 + n*t_delta

    Args:
    times_retarded - array(float)[radius,time] - retared times for each detector
    time_samples - int - desired number of time samples
    window_width - float - fractional length of window unaltered by windowing function. Window function assumed symmetrical.
    t0 - float - fixed time for syncing
    t_delta - time difference for syncing

    Returns:
    array(float)[time_samples] - time samples that are inside window for all radius samples and synced to t0 and t_delta
    """
    t_min = np.min(times_retarded[ 0]) + 0.5*(1.0-window_width)*(times_retarded[ 0][-1]-times_retarded[ 0][0])
    t_max = np.max(times_retarded[-1]) - 0.5*(1.0-window_width)*(times_retarded[-1][-1]-times_retarded[-1][0])
    
    n_min = int(ceil((t_min-t0)/t_delta))
    n_max = int(floor((t_max-t0)/t_delta))
    
    time_samples = n_max - n_min + 1
    
    t_start = t0 + n_min*t_delta
    t_end = t0 + n_max*t_delta
    
    return np.linspace(t_start,t_end,num = time_samples)

def Weyl4Extrapolation(weyl4_data,
                       radii,
                       times_raw,
                       weyl4_extrapolation_samples,
                       window_type,
                       omega_cutoff,
                       poly_order=3,
                       radii_power=-2.0,
                       window_kwargs = None,
                       verbose=False):
    """
    Polynomial extrapolation and FFI of Weyl4 data.

    Args:
    weyl4_data - array(float) - raw Weyl4 data
    radii - array(float) - extraction radii
    times_raw - array(float) - global frame time samples
    weyl4_extrapolation_samples - array(float) - number of time samples for extrapolation
    window_type - str - type of windowing function to use
    omega_cutoff - float - cutoff angluar frequency to pass to Get_Omegas

    Kwargs:
    poly_order=3 - int - order of fitting polynomial to use for extrapolation
    radii_power=-2.0 - float - polynomial fitted as P(radius**radii_power)
    window_kwargs=None - dict - passed to windowing function
    verbose=False - bool - print messages to IO

    Returns:
    array(float) - extrapolation time samples
    array(float) - extrapolated strain data
    """
    weyl4_ma_scaled = np.zeros(weyl4_data.shape)
    for r_idx in range(len(radii)):
        weyl4_ma_scaled[r_idx,:,0] = radii[r_idx]*np.abs(-weyl4_data[r_idx,:,0] + 1.0j*weyl4_data[r_idx,:,1])
        weyl4_ma_scaled[r_idx,:,1] = np.unwrap(np.arctan2(weyl4_data[r_idx,:,1],-weyl4_data[r_idx,:,0]))
    
    times_retarded = np.zeros(times_raw.shape)
    
    for r_idx in range(len(radii)):
        times_retarded[r_idx] = times_raw[r_idx] - radii[r_idx]
    
    time_interpolation_min = np.min(times_retarded[0])
    time_interpolation_max = np.max(times_retarded[-1])
    times_interpolation = np.linspace(time_interpolation_min,time_interpolation_max,num=weyl4_extrapolation_samples)
    
    weyl4_ma_scaled_interpolation = np.zeros((len(radii),len(times_interpolation),2))
    
    for r_idx in range(len(radii)):
        mag_func_interp = interp.interp1d(times_retarded[r_idx],weyl4_ma_scaled[r_idx,:,0],kind="cubic")
        arg_func_interp = interp.interp1d(times_retarded[r_idx],weyl4_ma_scaled[r_idx,:,1],kind="cubic")
        weyl4_ma_scaled_interpolation[r_idx,:,0] = mag_func_interp(times_interpolation)
        weyl4_ma_scaled_interpolation[r_idx,:,1] = arg_func_interp(times_interpolation)
    
    for t_idx in range(len(times_interpolation)):
        weyl4_ma_scaled_interpolation[:,t_idx,1] = np.unwrap(weyl4_ma_scaled_interpolation[:,t_idx,1])
    
    extrap_xs = radii**radii_power
    weights = radii**-radii_power
    
    weyl4_mag_poly_coeffs = np.polyfit(extrap_xs,weyl4_ma_scaled_interpolation[:,:,0],poly_order,w=weights)
    weyl4_arg_poly_coeffs = np.polyfit(extrap_xs,weyl4_ma_scaled_interpolation[:,:,1],poly_order,w=weights)
    
    weyl4_cplx_extrap = weyl4_mag_poly_coeffs[-1,:] * np.exp(1.0j*weyl4_arg_poly_coeffs[-1,:])
    
    window_arr = GetWindowingFunction(window_type, weyl4_extrapolation_samples, **window_kwargs)
    weyl4_cplx_extrap_windowed = window_arr * weyl4_cplx_extrap
    
    fft_interp_mix  = np.fft.fftshift(np.fft.fft(weyl4_cplx_extrap_windowed))
    fft_interp_freq = np.fft.fftshift(np.fft.fftfreq(weyl4_extrapolation_samples,d=times_interpolation[1]-times_interpolation[0]))
    
    omegas = Get_Omegas(fft_interp_freq,omega_cutoff)
    
    fft_integ2 = -fft_interp_mix/(omegas**2)
    
    cplx_extrap = np.fft.ifft(np.fft.ifftshift(fft_integ2))
    
    return times_interpolation, cplx_extrap

def DoExtrapStrain(strain_data_scaled,retarded_times,extrap_times,radii,verbose=False,extrap_type="weighted_sum",poly_order=1):
    """
    Wrapper for strain based extrapolation.
    """
    if extrap_type=="weighted_sum":
        return WeightedSumExtrapStrain(strain_data_scaled,retarded_times,extrap_times,radii,verbose=verbose)
    elif extrap_type=="poly":
        return PolyExtrapStrain(strain_data_scaled,retarded_times,extrap_times,radii,order=poly_order,verbose=verbose)
    else:
        raise ValueError
        
def WeightedSumExtrapStrain(strain_data_scaled,retarded_times,extrap_times,radii,verbose=False):
    """
    Weighted sum extrapolation of strain.
    """
    denominator = 0.0
    cplx_extrap = np.zeros(len(extrap_times),dtype=complex)
    
    for i in range(len(radii)):
        if verbose: print("Working on radius " + str(radii[i]))
        func_real_interp = interp.interp1d(retarded_times[i],np.real(strain_data_scaled[i]), kind="cubic")
        func_imag_interp = interp.interp1d(retarded_times[i],np.imag(strain_data_scaled[i]), kind="cubic")
        cplx_extrap += (func_real_interp(extrap_times) + 1.0j*func_imag_interp(extrap_times)) * radii[i]
        denominator += radii[i]
    
    cplx_extrap /= denominator
    
    return cplx_extrap    

def PolyExtrapStrain(strain_data_scaled,retarded_times,extrap_times,radii,radii_power=-2.0,order=1,verbose=False):
    """
    Polynomial extrapolation of strain data.
    """
    #cplx_interp = np.zeros((len(radii),len(extrap_times)),dtype=complex)
    mag_interp = np.zeros((len(radii),len(extrap_times)))
    arg_interp = np.zeros((len(radii),len(extrap_times)))

    for i in range(len(radii)):
        if verbose: print("Working on radius " + str(radii[i]))
        #func_real_interp = interp.interp1d(retarded_times[i],np.real(strain_data_scaled[i]), kind="cubic")
        #func_imag_interp = interp.interp1d(retarded_times[i],np.imag(strain_data_scaled[i]), kind="cubic")
        func_mag_interp = interp.interp1d(retarded_times[i],np.abs(strain_data_scaled[i]), kind="cubic")
        func_arg_interp = interp.interp1d(retarded_times[i],np.unwrap(np.arctan2(np.imag(strain_data_scaled[i]),np.real(strain_data_scaled[i]))), kind="cubic")
        #cplx_interp[i] = (func_real_interp(extrap_times) + 1.0j*func_imag_interp(extrap_times))
        #mag_interp[i] = np.abs(cplx_interp[i])
        #arg_interp[i] = np.arctan2(np.imag(cplx_interp[i]),np.real(cplx_interp[i]))
        mag_interp[i] = func_mag_interp(extrap_times)
        arg_interp[i] = func_arg_interp(extrap_times)
    
    for t_idx in range(len(extrap_times)):
        arg_interp[:,t_idx] = np.unwrap(arg_interp[:,t_idx])
    
    np.save("arg_interp.npy",arg_interp)
    np.save("extrap_times.npy",extrap_times)
    np.save("mag_interp.npy",mag_interp)
    np.save("radii.npy",radii)
    
    weights = radii**-radii_power
    
    if verbose: print("Fitting polynomials")
    #mag_interp = np.abs(cplx_interp)
    #arg_interp = np.arctan2(np.imag(cplx_interp),np.real(cplx_interp))
    mag_poly_coeffs = np.polyfit(radii**radii_power,mag_interp[:,:],order,w=weights)
    arg_poly_coeffs = np.polyfit(radii**radii_power,arg_interp[:,:],order,w=weights)
    
    cplx_extrap = mag_poly_coeffs[-1,:] * np.exp(1.0j*arg_poly_coeffs[-1,:])
    
    return cplx_extrap

def ExtractSignalFromWindow(data_times,data_strain,window_start,window_finish,new_times=None):
    """
    Extract clean signal from data that has had window function applied to it.

    Args:
    data_times - array(float) - time samples
    data_strain - array(complex) - physical strain data
    window_start - float - fractional length of start ramp
    window_finish - float - fractional beginning of end ramp
    
    Kwargs:
    new_times=None - array(float) - if present, interpolation is performed to sample data at these times

    Returns:
    array(float) - extracted sample times
    array(complex) - extracted data
    """
    index_count = len(data_times)
    extract_start = int(ceil(index_count*window_start))
    extract_end = int(floor(index_count*window_finish))
    
    return data_times[extract_start:extract_end], data_strain[extract_start:extract_end]
        

def GetMagArg(strain_data,times):
    """
    Magnitue and time derivative of arguement of physical strain.

    Args:
    strain_data - array(complex) - physical strain
    times - array(float) - sample times

    Returns:
    array(float) - magnitude of physical strain
    array(float) - time derivate of arguement of physical strain
    """
    mag_extrap = np.abs(strain_data)
    arg_extrap = np.arctan2(np.imag(strain_data),np.real(strain_data))
    arg_extrap_smooth = np.unwrap(arg_extrap)
    dargdt_extrap = np.gradient(arg_extrap_smooth)/np.gradient(times)
    
    return mag_extrap, dargdt_extrap

def GetInstantaneousFrequency(strain_data,times):
    """
    Instantaneous frequency of strain assuming amplitude of h_+ and h_x are equal.

    Args:
    strain_data - array(complex) - physical strain
    times - array(float) - sample times

    Returns:
    array(float) - instantaneous frequency of physical strain
    """
    return np.gradient(np.unwrap(np.arctan2(np.imag(strain_data),np.real(strain_data))))/(np.gradient(times)*2.0*np.pi)

def PowerlawPsdGaussian(exponent, samples, fmin=0):
    """Gaussian (1/f)**beta noise.
    
    Based on the algorithm in:
    Timmer, J. and Koenig, M.: 
    On generating power law noise. 
    Astron. Astrophys. 300, 707-710 (1995)
    
    Normalised to unit variance
    
    Parameters:
    -----------
    
    exponent : float
        The power-spectrum of the generated noise is proportional to
        
        S(f) = (1 / f)**beta
        flicker / pink noise:   exponent beta = 1
        brown noise:            exponent beta = 2
        
        Furthermore, the autocorrelation decays proportional to lag**-gamma 
        with gamma = 1 - beta for 0 < beta < 1.
        There may be finite-size issues for beta close to one.
    
    samples : int
        number of samples to generate
    
    fmin : float, optional
        Low-frequency cutoff.
        Default: 0 corresponds to original paper. It is not actually
        zero, but 1/samples.
        
    Returns
    -------
    out : array
        The samples.
    Examples:
    ---------
    # generate 1/f noise == pink noise == flicker noise
    >>> import colorednoise as cn
    >>> y = cn.powerlaw_psd_gaussian(1, 5)
    """
    
    # frequencies (we asume a sample rate of one)
    f = np.fft.fftfreq(samples)
    
    # scaling factor for all frequencies 
    ## though the fft for real signals is symmetric,
    ## the array with the results is not - take neg. half!
    s_scale = abs(np.concatenate([f[f<0], [f[-1]]]))
    ## low frequency cutoff?!?
    if fmin:
        ix = sum(s_scale>fmin)
        if ix < len(f):
            s_scale[ix:] = s_scale[ix]
    s_scale = s_scale**(-exponent/2.)
    
    # scale random power + phase
    sr = s_scale * np.random.normal(size=len(s_scale))
    si = s_scale * np.random.normal(size=len(s_scale))
    if not (samples % 2): si[0] = si[0].real

    s = sr + 1J * si
    # this is complicated... because for odd sample numbers,
    ## there is one less positive freq than for even sample numbers
    s = np.concatenate([s[1-(samples % 2):][::-1], s[:-1].conj()])

    # time series
    y = np.fft.ifft(s).real

    return y / np.std(y)