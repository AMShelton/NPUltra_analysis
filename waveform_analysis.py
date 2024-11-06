import os 
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def get_amps(unit):
    '''Calculate the largest amplitude signal from trough to peak for each channel.'''
    amps = []
    for chan in unit:
        trough_idx = np.where(chan == np.min(chan))[0][0]
        peak_idx = np.where(chan[trough_idx:] == chan[trough_idx:].max())[0][0] + trough_idx
        amp = (chan[peak_idx] + abs(chan[trough_idx]))
        amps.append(amp)
    return np.array(amps)

def get_wf_fft(waveform, sampling_rate=30000,height=50):
    '''Compute FFT and find significant peaks in frequency domain.'''
    F = []
    fft = []
    for chan in waveform:
        fft_data = np.fft.fft(chan)
        freqs = np.fft.fftfreq(len(fft_data), d=1/sampling_rate)
        F.append(freqs)
        fft.append(fft_data)
    averages = [np.mean(waveform, axis=0), np.mean(np.abs(fft), axis=0)]
    peaks = find_peaks(averages[1], height=height)
    return peaks

def calculate_spike_width(waveform):
    '''Calculate the spike width as the time between the trough and the peak.'''
    widths = []
    for chan in waveform:
        trough_idx = np.where(chan == np.min(chan))[0][0]
        peak_idx = np.where(chan[trough_idx:] == chan[trough_idx:].max())[0][0] + trough_idx
        width = peak_idx - trough_idx
        widths.append(width)
    return np.array(widths)

def calculate_spike_symmetry(waveform):
    '''Evaluate spike symmetry as the ratio of time from trough to peak vs peak to next trough.'''
    symmetries = []
    for chan in waveform:
        trough_idx = np.where(chan == np.min(chan))[0][0]
        peak_idx = np.where(chan[trough_idx:] == chan[trough_idx:].max())[0][0] + trough_idx
        next_trough_idx = np.where(chan[peak_idx:] == chan[peak_idx:].min())[0][0] + peak_idx
        if next_trough_idx > peak_idx:
            symmetry = (peak_idx - trough_idx) / (next_trough_idx - peak_idx)
            symmetries.append(symmetry)
    return np.array(symmetries)

def calculate_noise_level(waveform, pre_spike_window=10, post_spike_window=10):
    '''Calculate noise level by analyzing the variance in pre- and post-spike periods.'''
    noise_levels = []
    for chan in waveform:
        trough_idx = np.where(chan == np.min(chan))[0][0]
        pre_spike_noise = np.var(chan[max(0, trough_idx - pre_spike_window):trough_idx])
        post_spike_noise = np.var(chan[trough_idx + 1: min(len(chan), trough_idx + post_spike_window)])
        average_noise = (pre_spike_noise + post_spike_noise) / 2
        noise_levels.append(average_noise)
    return np.array(noise_levels)

def calculate_snr(waveform,axis=-1):
    signal = np.median(waveform, axis=axis)
    noise = waveform - signal[..., np.newaxis]
    snr = np.mean(signal, axis=axis) / np.std(noise, axis=axis)
    return snr

def calculate_rms_noise(waveform):
    '''Calculate the short and long rms noise for a given unit waveform.'''
    srms = np.mean([np.sqrt(sum([(x)**2 for i,x in enumerate(chan) if 0<=i<=15])/15) for chan in waveform])
    lrms = np.mean([np.sqrt(sum([(x)**2 for x in chan])/len(chan)) for chan in waveform])

    return srms,lrms

def calculate_spike_corr(waveform):
    """
    Calculate the pairwise correlation coefficients between channels for a given unit's waveform.
    
    Parameters:
    waveform : array_like
        2D array where each row corresponds to a channel and each column corresponds to a time point.

    Returns:
    correlations : array_like
        2D array containing the pairwise correlation coefficients between channels.
    mean_correlation : float
        The mean of the pairwise correlation coefficients.
    """
    # Calculate the correlation matrix
    correlations = np.corrcoef(waveform)
    
    # Extract the upper triangle (excluding the diagonal) to avoid duplicate pairs
    upper_triangle_indices = np.triu_indices_from(correlations, k=1)
    upper_triangle_correlations = correlations[upper_triangle_indices]
    
    # Compute the mean correlation coefficient
    mean_correlation = np.mean(upper_triangle_correlations)
    
    return correlations, mean_correlation

def downsample_npultra(unit,probe_type='1.0'):
    '''
    Given a unit array of shape (n_chan,t) in the NP Ultra, perform regular grid interpolation to find the multi-channel waveform with a NP 1.0 or NP 2.0 configuration.

    unit = 2d array of NP Ultra unit electrophysiological mean waveform of shape (n_chan,t).
    probe_type = string, determines points for interpolation to NP 1.0 or NP 2.0 configuration.
    '''
    from scipy.interpolate import RegularGridInterpolator as rgi
    
    ultra_data = unit.reshape(48,8,unit.shape[1])

    x = np.arange(ultra_data.shape[0])
    y = np.arange(ultra_data.shape[1])
    z = np.arange(ultra_data.shape[2])

    func = rgi((x,y,z), ultra_data,method='linear')

    if probe_type=='1.0':
        
        cols_coords = [0.5,2.5,4.5,6.5]
        row_coords = np.arange(0.5,48,4)

        X,Y = np.meshgrid(cols_coords,row_coords)

        X1 = X[::2,::2]
        X2 = X[::2,1::2]
        Y1 = Y[::2,:2]
        Y2 = Y[1::2,:2]

        col1_coords = np.column_stack([Y1[:,0],X1[:,0]])
        col2_coords = np.column_stack([Y2[:,0],X2[:,0]])
        col3_coords = np.column_stack([Y1[:,1],X1[:,1]])
        col4_coords = np.column_stack([Y2[:,1],X2[:,1]])

        points = []

        for i,p in enumerate(col1_coords):
            points.append(tuple(p))
            points.append(tuple(col3_coords[i]))
            points.append(tuple(col2_coords[i]))
            points.append(tuple(col4_coords[i]))

        V = np.asarray([[func([p[0],p[1],i])[0] for i in np.arange(0,82)] for p in points])

    if probe_type=='2.0':
        cols_coords = [1,5.0]
        row_coords = np.arange(1,48,3.5)


        X,Y = np.meshgrid(cols_coords,row_coords)

        X1 = X[:,0]
        X2 = X[:,1]
        Y1 = Y[:,0]
        Y2 = Y[:,1]

        col1_coords = np.column_stack([Y1,X1])
        col2_coords = np.column_stack([Y2,X2])

        points = []

        for i,p in enumerate(col1_coords):
            points.append(tuple(p))
            points.append(tuple(col2_coords[i]))

        V = []

        for p in points:
            vals = []

            for i in np.arange(0,82):
                wf = ([p[0],p[1],i])

                vals.append(func(wf)[0])
            V.append(vals)
    return V

def get_single_chan_features(unit):
    '''Given a 2D array of waveforms of shape n samples x time, return 1D arrays of values of shape n samples.'''
    from scipy.stats import linregress
    trough_idx = np.where(unit==np.min(unit))[0][0]
    # try:
    #     peak_idx = find_peaks(unit[trough_idx:])[0][0]+trough_idx
    #     prepeak_idx = np.where(unit[:trough_idx]==np.max(unit[:trough_idx]))[0][0]
    # except:
    peak_idx = np.where(unit[trough_idx:]==np.max(unit[trough_idx:]))[0][0]+trough_idx
    trough_h = unit[trough_idx]
    peak_h = unit[peak_idx]
    amp = (peak_h+abs(trough_h))
    try:
        prepeak_idx = np.where(unit[:trough_idx]==np.max(unit[:trough_idx]))[0][0]
        prepeak_h = unit[prepeak_idx]
        prePTR = prepeak_h/abs(trough_h)
    except:
        prepeak_h = 1
        prePTR = prepeak_h/abs(trough_h)
        
    dur = (peak_idx-trough_idx)/30
    
    PTR = peak_h/abs(trough_h)
    
    try:
        repol_slope = linregress(np.linspace(trough_idx,trough_idx+5,5),unit[trough_idx:trough_idx+5])[0]*(30)
    except:
        repol_slope = 0

    try:
        recov_slope = linregress(np.linspace(peak_idx,peak_idx+5,5),unit[peak_idx:peak_idx+5])[0]*(30)
    except:
        recov_slope = 0
    
    return amp,dur,PTR,prePTR,repol_slope,recov_slope