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
    srms = np.mean([np.sqrt(sum([(x)**2 for i,x in enumerate(chan) if 0<=i<=15])/len(15)) for chan in waveform])
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