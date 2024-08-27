from waveform_analysis import *
from spike_time_analysis import *
from utils import *

def main(waveform_file, spike_times_file):
    # Load data
    waveforms = load_waveform_data(waveform_file)
    spike_times = load_spike_time_data(spike_times_file)
    
    # Process waveforms
    for i, waveform in enumerate(waveforms):
        amps = get_amps(waveform)
        fft_peaks = get_wf_fft(waveform)
        spike_widths = calculate_spike_width(waveform)
        symmetries = calculate_spike_symmetry(waveform)
        noise_levels = calculate_noise_level(waveform)
        
        # Combine and store waveform metrics for analysis
        waveform_metrics = {
            'Amplitude': amps,
            'FFT Peaks': fft_peaks,
            'Spike Width': spike_widths,
            'Symmetry': symmetries,
            'Noise': noise_levels,
        }
        print(f'Waveform {i} metrics: {waveform_metrics}')
        
    # Process spike times
    for i, spike_times_unit in enumerate(spike_times):
        firing_rate = calculate_firing_rate(spike_times_unit)
        isi = calculate_isi(spike_times_unit)
        burst_count, _ = detect_bursts(spike_times_unit)
        cv_isi = calculate_spike_train_regularity(isi)
        refractory_violations = count_refractory_violations(isi)
        
        # Combine and store spike time metrics for analysis
        spike_time_metrics = {
            'Firing Rate': firing_rate,
            'Mean ISI': np.mean(isi),
            'Burst Count': burst_count,
            'CV of ISI': cv_isi,
            'Refractory Violations': refractory_violations,
        }
        print(f'Spike Time {i} metrics: {spike_time_metrics}')
    
    # Further analysis or classification based on metrics
    # ...

if __name__ == "__main__":
    waveform_file = 'path_to_waveform_file.npy'
    spike_times_file = 'path_to_spike_times_file.pkl'
    main(waveform_file, spike_times_file)