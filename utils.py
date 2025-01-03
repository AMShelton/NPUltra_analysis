import pickle
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from scipy.signal import butter,filtfilt,find_peaks

def load_waveform_data(filepath):
    '''Load waveform data from a numpy file.'''
    return np.load(filepath)

def load_spike_time_data(filepath):
    '''Load spike time data from a pickle file.'''
    with open(filepath, 'rb') as f:
        return pickle.load(f)
    
def butter_filter(data, cutoff, fs, order, btype=None):
    '''Filter an input waveform or multi-channel timeseries array.
    btype: {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}, optional'''
    nyq = 0.5 * fs 
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    y = filtfilt(b, a, data)
    return y

def tolerant_mean(arrs):

    '''Allows for averaging across axis=1 in a 2D ragged array. Averaging does not include arrays with len < current idx'''

    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)

def process_waveforms(waveform_arr, settings_path, probe_label='ProbeA', probe_type='Neuropixels Ultra (Switchable)', shape=(48,8,90), site_size=6,car=True):
    '''
    Rearrange waveform channel index for each unit in a Neuropixels Ultra probe recording to match that of a recording settings.xml or pre-defined site coordinates .csv.

    waveform_arr: array_like
        The input data array of average waveforms, arranged as units x channels x sample
    settings_path: str
        Can be a Windows Path object. Path directing to either a settings.xml file or coords.csv file.
    probe_label: label of the probe to process.
    probe_type: {'Neuropixels Ultra','Neuropixels Ultra (Switchable)',''}
    '''

    corrected_waveforms = []  # Initialize the output list

    if probe_type == 'Neuropixels Ultra (Switchable)' and shape[:2] != (48,8):
        print(f"Processing waveforms for {probe_label} in {shape[0]} x {shape[1]} configuration")
        settings = pd.read_csv(settings_path)

        unique_xpos = sorted(settings['xpos'].unique())
        unique_ypos = sorted(settings['ypos'].unique())

        xpos_mapping = {value: index % shape[1] for index, value in enumerate(unique_xpos)}
        ypos_mapping = {value: index % shape[0] for index, value in enumerate(unique_ypos)}

        for neuron_index in range(waveform_arr.shape[0]):
            neuron_waveform = waveform_arr[neuron_index]
            reshaped_waveform = np.zeros((shape[0], shape[1], shape[2]))

            for _, row in settings.iterrows():
                channel_index = row['channel']
                xpos = row['xpos']
                ypos = row['ypos']

                ypos_index = ypos_mapping[ypos]
                xpos_index = xpos_mapping[xpos]

                reshaped_waveform[ypos_index, xpos_index, :] = neuron_waveform[channel_index, :]

            corrected_waveforms.append(reshaped_waveform.reshape(shape[0]*shape[1],shape[2]))
    
    elif 'Neuropixels Ultra' in probe_type and shape[:2] == (48,8):
        print(f"Processing waveforms for {probe_label} in 48 x 8 configuration")
        tree = ET.parse(settings_path)
        root = tree.getroot()

        channel_positions = []

        for probe in root.findall(f".//NP_PROBE[@custom_probe_name='{probe_label}']"):
            # print(probe.attrib['custom_probe_name'])
            x_pos = probe.find('ELECTRODE_XPOS')
            y_pos = probe.find('ELECTRODE_YPOS')

            for channel in x_pos.attrib:
                index = int(channel.replace('CH', ''))
                x = int(x_pos.attrib[channel]) // site_size  # Convert microns to pixels
                y = int(y_pos.attrib[channel]) // site_size  # Convert microns to pixels
                channel_positions.append((index, x, y))

        channel_positions.sort(key=lambda pos: (pos[2], pos[1]))
        pos = [pos[0] for pos in channel_positions]

        for wf in waveform_arr:
            reshaped_wf = wf[pos]
            corrected_waveforms.append(reshaped_wf)
    
    corrected_waveforms = np.array(corrected_waveforms)

    if car:
        for i,wf in enumerate(corrected_waveforms):
            cm = np.median(wf,axis=0)
            cm_wf = wf-cm
            corrected_waveforms[i] = cm_wf

    return corrected_waveforms

def save_figure(fig, fname, formats = ['.pdf'],transparent=False,dpi=300,facecolor=None,**kwargs):
    import matplotlib as mpl
    mpl.rcParams['pdf.fonttype'] = 42

    if 'size' in kwargs.keys():
        fig.set_size_inches(kwargs['size'])

    elif 'figsize' in kwargs.keys():
        fig.set_size_inches(kwargs['figsize'])
    for f in formats:
        fig.savefig(fname + f, transparent = transparent,dpi=dpi)