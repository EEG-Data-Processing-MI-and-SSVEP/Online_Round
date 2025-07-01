import os, sys, mne, torch
from dotenv import load_dotenv
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from scipy import signal

# Get the directory of the current file
current_file_dir = os.path.dirname(__file__)

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(current_file_dir, '..'))

# Add the parent directory to Python path
sys.path.append(parent_dir)

env_path = os.path.join(parent_dir, '.env')
load_dotenv(env_path)

class SSVEPDataset(Dataset):
    """Enhanced SSVEP dataset loader with frequency feature extraction"""
    
    # SSVEP frequencies for each class
    FREQUENCIES = {
        'Left': 10,
        'Right': 13,
        'Forward': 7,
        'Backward': 8
    }
    
    def __init__(self, csv_path, task='SSVEP', type='Train', eeg_reference='average', transform=None):
        """
        Args:
            csv_path: Path to metadata CSV
            task: Task type (only SSVEP supported)
            eeg_reference: Reference method for EEG
            transform: Optional transforms to be applied
        """

        self.base_path = os.getenv('DATA_BASE_DIR')
        self.metadata = pd.read_csv(csv_path)
        self.eeg_reference = eeg_reference
        self.transform = transform
        self.type = type
        if task:
            self.metadata = self.metadata[self.metadata['task'] == task]
            
        # Create label encoding
        print(sorted(self.FREQUENCIES.keys()))
        self.classes = sorted(self.FREQUENCIES.keys())
        self.label_encoding = {label: i for i, label in enumerate(self.classes)}
        
        # Precompute frequency bands of interest
        self.freq_bands = {
            label: (freq-1, freq+1) for label, freq in self.FREQUENCIES.items()
        }
        
        # Sampling rate (Hz)
        self.sfreq = 250
        
    def apply_reference(self, raw):
        """Apply EEG referencing"""
        try:
            if self.eeg_reference == 'average':
                raw.set_eeg_reference('average', verbose=False)
            elif self.eeg_reference in raw.ch_names:
                raw.set_eeg_reference([self.eeg_reference], verbose=False)
            elif self.eeg_reference is not None:
                raise ValueError(f"Invalid EEG reference: {self.eeg_reference}")
        except Exception as e:
            raise RuntimeError(f"EEG referencing failed: {e}")
        return raw
    
    def extract_frequency_features(self, eeg_data):
        """Extract power in SSVEP frequency bands"""
        n_channels, n_samples = eeg_data.shape
        features = np.zeros((n_channels, len(self.freq_bands)))
        
        for i, (label, (low, high)) in enumerate(self.freq_bands.items()):
            # Compute power spectral density using Welch's method
            freqs, psd = signal.welch(eeg_data, fs=self.sfreq, nperseg=min(256, n_samples))
            
            # Find indices of frequency band
            idx = np.logical_and(freqs >= low, freqs <= high)
            
            # Compute average power in band for each channel
            features[:, i] = np.mean(psd[:, idx], axis=1)
            
        return features
    
    def normalize(self, data):
        """Normalize data channel-wise"""
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        return (data - mean) / (std + 1e-6)
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        id_num = row['id']
        dataset = 'train' if id_num <= 4800 else 'validation' if id_num <= 4900 else 'test'
        
        # Load EEG data
        eeg_path = os.path.join(
            self.base_path, row['task'], dataset,
            row['subject_id'], str(row['trial_session']), "EEGdata.csv"
        )
        
        df = pd.read_csv(eeg_path)
        df["Time"] -= df["Time"].iloc[0]
        
        # Extract trial data
        trial_num = int(row['trial'])
        samples_per_trial = 1750
        start_idx = (trial_num - 1) * samples_per_trial
        end_idx = start_idx + samples_per_trial
        df = df.iloc[start_idx:end_idx]
        
        # EEG processing
        eeg_channels = ['PO7', 'OZ', 'PO8']
        if not all(ch in df.columns for ch in eeg_channels):
            raise ValueError(f"Missing required EEG channels in file: {eeg_path}")
        
        eeg_data = df[eeg_channels].values.T
        info = mne.create_info(ch_names=eeg_channels, sfreq=self.sfreq, ch_types='eeg')
        raw = mne.io.RawArray(eeg_data, info, verbose=False)
        
        # Apply notch filter at 50Hz and bandpass filter in SSVEP range
        raw.notch_filter(freqs=50, verbose=False)
        raw.filter(l_freq=6, h_freq=30, verbose=False)  # Wider band to capture all frequencies
        
        # Apply referencing
        raw = self.apply_reference(raw)
        
        # Get filtered data
        eeg_filtered = raw.get_data()
        
        # Extract frequency features
        freq_features = self.extract_frequency_features(eeg_filtered)
        
        # Normalize
        eeg_normalized = self.normalize(eeg_filtered).astype(np.float32)
        freq_features = self.normalize(freq_features).astype(np.float32)
        
        # Motion data
        motion_channels = ['AccX', 'AccY', 'AccZ', 'Gyro1', 'Gyro2', 'Gyro3']
        motion_data = df[motion_channels].values.T
        motion_normalized = self.normalize(motion_data).astype(np.float32)
        
        # Convert to tensors
        eeg_tensor = torch.from_numpy(eeg_normalized)
        freq_tensor = torch.from_numpy(freq_features)
        motion_tensor = torch.from_numpy(motion_normalized)    
        if self.transform:
            eeg_tensor = self.transform(eeg_tensor)

        if self.type.lower() == 'test':
            return eeg_tensor, freq_tensor, motion_tensor, row['id']
        else:
            label_tensor = torch.tensor(self.label_encoding[row['label']], dtype=torch.long)
            return eeg_tensor, freq_tensor, motion_tensor, label_tensor
