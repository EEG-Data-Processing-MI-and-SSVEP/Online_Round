import os
import mne
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from scipy import signal
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

class SSVEPDataset(Dataset):
    """Enhanced SSVEP dataset loader with frequency feature extraction"""
    
    # SSVEP frequencies for each class
    FREQUENCIES = {
        'Left': 10,
        'Right': 13,
        'Forward': 7,
        'Backward': 8
    }
    
    def __init__(self, csv_path, task='SSVEP', eeg_reference='average', transform=None):
        """
        Args:
            csv_path: Path to metadata CSV
            task: Task type (only SSVEP supported)
            eeg_reference: Reference method for EEG
            transform: Optional transforms to be applied
        """
        self.base_path = '/kaggle/input/mtcaic3'
        self.metadata = pd.read_csv(csv_path)
        self.eeg_reference = eeg_reference
        self.transform = transform
        
        if task:
            self.metadata = self.metadata[self.metadata['task'] == task]
            
        # Create label encoding
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
        
        # --- Label ---
        if 'label' in row:
            label_tensor = torch.tensor(self.label_encoding[row['label']], dtype=torch.long)
        else:
            label_tensor = torch.tensor(-1, dtype=torch.long)  # Placeholder for test data
        
        if self.transform:
            eeg_tensor = self.transform(eeg_tensor)
            
        return eeg_tensor, freq_tensor, motion_tensor, label_tensor, row['id']
     

class EnhancedSSVEPModel(nn.Module):
    """Enhanced model for SSVEP classification with frequency attention"""
    
    def __init__(self, num_classes=4, eeg_channels=3, motion_channels=6, freq_bands=4):
        super(EnhancedSSVEPModel, self).__init__()
        
        # EEG branch - deeper architecture with residual connections
        self.eeg_branch = nn.Sequential(
            # First block
            nn.Conv1d(eeg_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.MaxPool1d(2),  # 875
            
            # Second block with residual
            ResidualBlock(64, 128, kernel_size=5, stride=2, downsample=True),  # 438
            
            # Third block
            ResidualBlock(128, 256, kernel_size=5, stride=2, downsample=True),  # 219
            
            # Fourth block
            ResidualBlock(256, 512, kernel_size=5, stride=2, downsample=True),  # 110
            
            # Attention pooling
            AttentionPooling(512),
            
            # Final projection
            nn.Linear(512, 256)
        )
        
        # Rest of the model remains the same...
        self.freq_branch = nn.Sequential(
            nn.Linear(eeg_channels * freq_bands, 128),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64)
        )
        
        self.motion_branch = nn.Sequential(
            nn.Conv1d(motion_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.MaxPool1d(2),  # 875
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.MaxPool1d(2),  # 438
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 + 64 + 128, 512),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, eeg, freq, motion):
        # EEG branch
        eeg_features = self.eeg_branch(eeg)  # [B, 256]
        
        # Frequency branch
        batch_size = freq.size(0)
        freq_features = self.freq_branch(freq.view(batch_size, -1))  # [B, 64]
        
        # Motion branch
        motion_features = self.motion_branch(motion).squeeze(-1)  # [B, 128]
        
        # Combine features
        combined = torch.cat([eeg_features, freq_features, motion_features], dim=1)
        logits = self.classifier(combined)
        
        return logits


class ResidualBlock(nn.Module):
    """Residual block with downsampling"""
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, downsample=False):
        super(ResidualBlock, self).__init__()
        self.downsample = downsample
        
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, 
                             kernel_size=kernel_size, 
                             padding=padding, 
                             stride=stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.elu = nn.ELU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, 
                             kernel_size=kernel_size, 
                             padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        if downsample or in_channels != out_channels:
            self.downsample_conv = nn.Conv1d(in_channels, out_channels, 
                                           kernel_size=1, 
                                           stride=stride)
            self.downsample_bn = nn.BatchNorm1d(out_channels)
        else:
            self.downsample_conv = None
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample_conv is not None:
            residual = self.downsample_conv(residual)
            residual = self.downsample_bn(residual)
            
        out += residual
        out = self.elu(out)
        
        return out


class AttentionPooling(nn.Module):
    """Attention-based temporal pooling"""
    def __init__(self, channels):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels, channels//8, kernel_size=1),
            nn.ELU(),
            nn.Conv1d(channels//8, 1, kernel_size=1),
            nn.Softmax(dim=2)
        )
        
    def forward(self, x):
        # x shape: [B, C, T]
        weights = self.attention(x)  # [B, 1, T]
        out = torch.sum(x * weights, dim=2)  # [B, C]
        return out