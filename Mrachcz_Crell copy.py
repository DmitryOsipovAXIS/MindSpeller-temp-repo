import os
from flask import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
import scipy.signal
from scipy.interpolate import griddata
from scipy.stats import kurtosis
from scipy import stats
import pywt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA # o semplicemente LDA
from sklearn.metrics import silhouette_score
import math
from statsmodels.stats.multitest import multipletests

# extract MRCP as per Crell et al. 
# epoching: B) compared to baseline rec.
# feature selected as per Mrachcz
# delta and stat relecance as per Prem

# --- CONFIGURATION ---
CHANNELS = ["Channel1", "Channel2", "Channel3", "Channel4", "Channel5", "Channel6"]
DATA_DIR = os.path.join(os.path.dirname(__file__),"data")
RUN_FILES = ["cued_run_6.csv"]  # Your 3 cued calibration files
FS = 500 #Hz
#FS_LOW = 10 #Hz -> Freq after downsampling (at the end of preprocessing)
HAND = 'rx'

# --- Preprocessing ---
def load_data(csv_path, channels=CHANNELS, fs = FS):
    # 1. Load EEG data (Assuming tab-separated format from your collection process)
    df = pd.read_csv(csv_path, sep='\t')
    eeg_data = df[channels].values.T  # Shape: (channels, samples)

    if 'baseline' in csv_path:
            cue_indices =  None
    else:
        json_path = csv_path.replace('.csv', '.json').replace('cued_run','labels/cued_run')
    
        # 2. Load cue timestamps from the JSON file
        with open(json_path, 'r') as f:
            cue_timestamps = np.array(json.load(f))
            
        cue_indices = np.array(cue_timestamps * FS, dtype=int)  
        if "single_cue" not in json_path:
            cue_indices = cue_indices - cue_indices[0]

    return eeg_data, cue_indices

def preprocess_eeg(eeg_raw, sfreq=FS):
    # removes DC, Nottch filtering, BP Butterwhorth (0.5 - 70 Hz), 
    # ICA + motor component selection, LP 3.5Hz, downsaple to 10Hz,
    # Common Average Referencing (CAR)

    # 1. Remove DC offset 
    eeg_raw=eeg_raw - np.mean(eeg_raw, axis=1, keepdims=True)

    # 2 Notch Filter
    b_notch, a_notch = scipy.signal.iirnotch(50, 30, fs=sfreq)
    eeg_raw = scipy.signal.filtfilt(b_notch, a_notch, eeg_raw, axis=1)

    # 3. 4th-order Butterworth Bandpass Filter (0.5 - 70 Hz)
    sos_bp = scipy.signal.butter(4, [0.5, 70], 'bp', fs=sfreq, output='sos')
    eeg_filtered = scipy.signal.sosfiltfilt(sos_bp, eeg_raw, axis=1)

    # 4. ICA
    ica = FastICA(n_components=6, algorithm='parallel', whiten='unit-variance', max_iter=200, random_state=0)
    ica_components = ica.fit_transform(eeg_filtered.T) # shape (samples, 6)
    mixing_matrix = ica.mixing_
    unmixing_matrix = ica.components_

    # Auto-select motor component
    trg = 2 if HAND == 'rx' else 3

    # normalize weigths from 0 to 1 for each source
    weights_all = np.array([(np.abs(mixing_matrix[:, i]) - np.min(np.abs(mixing_matrix[:, i]))) / 
                            (np.max(np.abs(mixing_matrix[:, i])) - np.min(np.abs(mixing_matrix[:, i])) + 1e-9) 
                            for i in range(mixing_matrix.shape[1])])

    # compute number of active channels per source    
    active_ch = np.sum(weights_all >= 0.6, axis=1)

    # compute score of target channel for all the sources
    all_scores = np.array([w[trg] / (np.mean(np.delete(w, trg)) + 1e-9) for w in weights_all])

    cx_scores = np.zeros(mixing_matrix.shape[1])

    #  --- Waterfall logic ---

    # TIER 1: Localized sources where target ch is active
    tier1_mask = (active_ch < 3) & (weights_all[:, trg] >= 0.6)
    # TIER 2: Spread sources where target ch is active
    tier2_mask = (active_ch == 3) & (weights_all[:, trg] >= 0.6)
    # TIER 3: Noisy sources where target ch is active
    tier3_mask = (active_ch == 4) & (weights_all[:, trg] >= 0.6)

    if np.any(tier1_mask):
        print("Localized sources found.")
        cx_scores[tier1_mask] = all_scores[tier1_mask]
    elif np.any(tier2_mask):
        print("Spread sources found.")
        cx_scores[tier2_mask] = all_scores[tier2_mask]
    elif np.any(tier3_mask):
        print("Only noisy sources found.")
        cx_scores[tier3_mask] = all_scores[tier3_mask]
    else:
        print("WARNING: Target ch is not active in any source.")
        # Fallback: take global scores
        cx_scores = weights_all[:, trg]

    best_idx = np.argmax(cx_scores)


    #best_idx = np.argmax(cx_scores)


    print(f"Selected ICA Component: {best_idx + 1} (C{trg+1} Score: {cx_scores[best_idx]:.2f})") 

    # Recompose dataset preserving only our extracted component to visualize reconstruction
    masked_sources = np.zeros_like(ica_components)
    masked_sources[:, best_idx] = ica_components[:, best_idx]
    recomposed_eeg = ica.inverse_transform(masked_sources) # Shape: (Samples, 6)
    recomposed_eeg = recomposed_eeg.T # Shape: (6, Samples)

    # 5. LP 3.5Hz
    sos_bp = scipy.signal.butter(4, 3.5, 'lp', fs=sfreq, output='sos')
    eeg_filtered = scipy.signal.sosfiltfilt(sos_bp, recomposed_eeg, axis=1)

    # 6. Downsample to 10Hz - NOTE: Skipped in this implementation
    FS_new = 10
    q = int(sfreq / FS_new)  
    sfreq = FS_new

    eeg_downsampled = scipy.signal.decimate(eeg_filtered, q, axis=1, ftype='iir')

    # 7. CAR
    eeg_downsampled = eeg_downsampled - np.mean(eeg_downsampled, axis=0, keepdims = True)

    mrcp = eeg_downsampled

    # --- end skip ---

    return eeg_filtered

def extract_epochs(mrep_data, cue_indices, tmin=-1, tmax=2, fs=FS):
    """
    Estrae le epoche attorno ai cue mantenendo la struttura (n_epochs, n_channels, n_times)
    """
    action_epochs_list = []
    samples_min = int(tmin * fs)
    samples_max = int(tmax * fs)

    occupied_mask = np.ones(mrep_data.shape[1], dtype=bool)

    for cue in cue_indices:
        start = cue + samples_min
        end = cue + samples_max
        # Controllo che l'epoca non esca dai limiti del segnale
        if start >= 0 and end < mrep_data.shape[1]:
            action_epochs_list.append(mrep_data[:, start:end])
            start = int(start - 1*FS)
            end = int(end + 1*FS)
            occupied_mask[start:end] = False
        
            
    # Array NumPy speculare alle richieste MNE: (epoche, canali, campioni)
    action_epochs_data = np.array(action_epochs_list)

    #Search for segments of the signal with no action
    rest_epochs_list = []
    current_chunk = []
    epoch_len_samples = samples_max - samples_min

    for i, mask in enumerate(occupied_mask):
        if mask:
            current_chunk.append(mrep_data[:, i])
        
        # If rest period ends or signal reaches the end
        if not mask or i == mrep_data.shape[1] - 1:
            if len(current_chunk) >= epoch_len_samples:
                chunk_data = np.column_stack(current_chunk) 

                # NOTE since we followed Crell et al.'s protocol in the collection of data
                # the trigger and the actions are not spaced evenely. Since we extracted 3s epochs
                # around the action the left outs do not have all the same size. Therefore we need 
                # to cut the rest epochs to all the same size.

                # Calculate how many full epochs fit into this rest block
                n_epochs, remainder = divmod(chunk_data.shape[1], epoch_len_samples)

                if any("single_cue" in path for path in RUN_FILES):
                    buffer = 6 * FS // epoch_len_samples    # 6s di buffer per le zone rest
                    n_epochs = n_epochs - buffer
                
                for e in range(n_epochs):
                    #start_idx =remainder + e * epoch_len_samples        # favors the end of the epoch (avoids leakage from end of action)
                    start_idx =e * epoch_len_samples
                    end_idx = start_idx + epoch_len_samples
                    rest_epochs_list.append(chunk_data[:, start_idx:end_idx])
            
            current_chunk = []
    
    rest_epochs_data = np.array(rest_epochs_list)
    
    return action_epochs_data, rest_epochs_data

def extract_epochs_baseline(baseline, tmin=-1, tmax=3, fs=FS):
    duration_sec = tmax - tmin  
    window = int(duration_sec * fs)
    
    total_samples = baseline.shape[1]
    num_epochs = total_samples // window
    
    epochs = []
    for i in range(num_epochs):
        start = i * window
        end = start + window
        epochs.append(baseline[:, start:end])
        
    # Returns 3D array: (n_epochs, n_channels, n_samples)
    return np.array(epochs)


# --- Feature Extraction ---
def extract_raw_features(epochs_data, fs=FS):
    """
    Extracts the following 4 features for each epoch and each channel.
    1. Kurtosis
    2. Marginal Sum of D3 Coeff from DWT with Db4 - with FS=500 : [31.25 Hz, 62.5 Hz]
    3. PSD 0 - 4 Hz
    4. Coefs from the fitting of a 3rd order polynomial
    
    Parameters:
    -----------
    epochs_data : np.ndarray
        3D array of shape (n_epochs, n_channels, n_times)
    fs : int
        Sampling frequency (default: FS)
        
    Returns:
    --------
    features_dict : dict
        Dictionary containing the extracted feature matrices.
        Each feature matrix has a shape of (n_epochs, n_channels).
        The polynomial coefficients matrix has a shape of (n_epochs, n_channels, 4).
    """
    if len(epochs_data.shape)==3:
        n_epochs, n_channels, n_times = epochs_data.shape
    elif len(epochs_data.shape) == 2:
        n_channels, n_times = epochs_data.shape
        n_epochs =1

    
    # Initialize the matrices that will hold the features (shape: epochs x channels)
    kurt_feats = np.zeros((n_epochs, n_channels))
    dwt_feats  = np.zeros((n_epochs, n_channels))
    psd_feats  = np.zeros((n_epochs, n_channels))
    
    # For the 3rd-order polynomial, we save the 4 coefficients: [w_3, w_2, w_1, w_0]
    poly_feats = np.zeros((n_epochs, n_channels, 4))
    
    # Normalized time vector between -1 and 1 to prevent numerical instability during regression
    t_norm = np.linspace(-1, 1, n_times)
    
    for e in range(n_epochs):
        for ch in range(n_channels):
            signal = epochs_data[e, ch, :]
            
            # 1. KURTOSIS EXTRACTION
            # fisher=True calculates excess kurtosis (where a normal distribution equals 0)
            kurt_feats[e, ch] = kurtosis(signal, fisher=True) 
            
            # 2. POWER SPECTRAL DENSITY (PSD) & 0-4 Hz SELECTION
            # We use the DPSS method described in MindSpeller work
            freqs, psd_values = compute_dpss_multitaper_psd(signal, fs=fs)
            idx_0_4 = (freqs >= 0) & (freqs <= 4)
            # Calculate the mean (or integrated) power within the Delta band (0-4Hz)
            psd_feats[e, ch] = np.mean(psd_values[idx_0_4])

            # 3. DISCRETE WAVELET TRANSFORM (DWT) - db4 level 5 (mu band)
            # Inspired by Mrachacz-Kersting et al., the feature is the "marginal sum" 
            # (sum over time) of the level 5 detail coefficients (D5).

            # first verify presence of muscle artifacts
            muscle_artifact = muscle_artifacts_check(freqs, psd_values)
            if not muscle_artifact:
                coeffs = pywt.wavedec(signal, 'db4', level=6)
                # wavedec returns list ordered as [cA6, cD6, cD5, cD4, cD3, cD2, cD1]
                cD5 = coeffs[2] 
                dwt_feats[e, ch] = np.sum(np.abs(cD5))
            else:
                dwt_feats[e, ch] = 0.0
            
            # 4. 3rd-ORDER POLYNOMIAL COEFFICIENTS (Macro-Morphology)
            # A 3rd-degree polynomial is described by: y = w_3*t^3 + w_2*t^2 + w_1*t + w_0
            # np.polyfit returns the weights ordered from highest degree to lowest: [w_3, w_2, w_1, w_0]
            poly_coefs = np.polyfit(t_norm, signal, deg=3)
            poly_feats[e, ch, :] = poly_coefs

    return {
        "kurtosis": kurt_feats,
        "dwt_marginal_sum": dwt_feats,
        "psd_delta": psd_feats,
        "poly_coefficients": poly_feats
    }


def main():

    print("=" * 80)
    print("Loading and Preprocessing Data...")
    print("=" * 80)
    


    # Data Files
    run_paths = [os.path.join(DATA_DIR, f) for f in RUN_FILES]
    for path in run_paths:
        if not os.path.exists(path):
            print(f"Error: Could not locate data file at '{path}'")
            return
    
        csv_path = path
        eeg_data, cue_indices = load_data(csv_path)

        filtered_eeg = preprocess_eeg(eeg_data)
        action_epochs, rest_epochs = extract_epochs(filtered_eeg, cue_indices)      

        features_action = extract_raw_features(action_epochs)       #{"feature" : (n_epochs, n_channels)}
        features_rest = extract_raw_features (rest_epochs)

    print("end")






if __name__ == "__main__":
    main()