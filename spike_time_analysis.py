import numpy as np

def calculate_firing_rate(spike_times, total_time=None):
    '''Calculate firing rate (spikes per second).'''
    if total_time is None:
        total_time = spike_times[-1]
    firing_rate = len(spike_times) / total_time
    return firing_rate

def calculate_isi(spike_times):
    '''Calculate inter-spike intervals (ISI).'''
    isi = np.diff(spike_times)
    return isi

def detect_bursts(spike_times, burst_threshold=0.01):
    '''Identify and count bursts in spike trains.'''
    isi = calculate_isi(spike_times)
    bursts = np.where(isi < burst_threshold)[0]
    return len(bursts), bursts

def calculate_spike_train_regularity(isi):
    '''Measure spike train regularity using CV of ISI.'''
    cv_isi = np.std(isi) / np.mean(isi)
    return cv_isi

def count_refractory_violations(isi, refractory_period=0.002):
    '''Count the number of refractory period violations.'''
    violations = np.sum(isi < refractory_period)
    return violations