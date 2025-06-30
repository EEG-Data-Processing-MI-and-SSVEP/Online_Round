import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import f1_score

import mne
from scipy import linalg
from mne.decoding import CSP
from scipy.signal import butter, filtfilt

class EEGDataset(Dataset):
    base_path = '/kaggle/input/mtcaic3'
    label_encoding = {'Left': 0, 'Right': 1}

    # Sensitivity values (replace with your device's actual values if different)
    gyro_scale = 0.02  # deg/s per LSB
    gyro_offset = 0
    acc_scale = 0.000598  # g per LSB
    acc_offset = 0

    def __init__(self, csv_path, task=None, segment_length=250, overlap=0.5):
        self.metadata = pd.read_csv(csv_path)
        self.task = task
        self.segment_length = segment_length
        self.overlap = overlap

        if self.task:
            self.metadata = self.metadata[self.metadata['task'] == self.task]

    def __len__(self):
        return len(self.metadata)

    def _butter_bandpass(self, lowcut=8.0, highcut=30.0, fs=250, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def _bandpass_filter(self, data, lowcut=8, highcut=30, fs=250, order=4):
        b, a = self._butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data, axis=0)
        return y

    def _spatial_filtering(self, eeg_values):
        return eeg_values  # placeholder for Laplacian if needed

    def _car_reference(self, eeg_values, cz_idx, pz_idx):
        car_ref = (eeg_values[cz_idx] + eeg_values[pz_idx]) / 2
        eeg_values = eeg_values - car_ref
        return eeg_values

    def _normalize(self, data):
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        return (data - mean) / std

    def _convert_gyro(self, gyro_values):
        # gyro_values: (3, n_samples)
        return (gyro_values - self.gyro_offset) * self.gyro_scale  # deg/s

    def _convert_acc(self, acc_values):
        # acc_values: (3, n_samples)
        return (acc_values - self.acc_offset) * self.acc_scale  # g

    def _get_eeg_data(self, row):
        id_num = row['id']
        if id_num <= 4800:
            dataset = 'train'
        elif id_num <= 4900:
            dataset = 'validation'
        else:
            dataset = 'test'

        eeg_path = f"{self.base_path}/{row['task']}/{dataset}/{row['subject_id']}/{row['trial_session']}/EEGdata.csv"
        eeg_data = pd.read_csv(eeg_path)
        trial_num = int(row['trial'])

        samples_per_trial = 2250 if row['task'] == 'MI' else 1750
        start_idx = (trial_num - 1) * samples_per_trial
        end_idx = start_idx + samples_per_trial - 1
        eeg_data = eeg_data.iloc[start_idx:end_idx + 1]
        return eeg_data

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        eeg_data = self._get_eeg_data(row)

        # EEG channels
        selected_channels = ['C3', 'CZ', 'C4', 'PZ']
        eeg_values = eeg_data[selected_channels].values.astype(np.float32).T  # (4, n_samples)
        eeg_values = self._bandpass_filter(eeg_values.T).T  # (4, n_samples)
        eeg_values = self._spatial_filtering(eeg_values)

        # CAR referencing using CZ and PZ, then remove them
        cz_idx = selected_channels.index('CZ')
        pz_idx = selected_channels.index('PZ')
        eeg_values = self._car_reference(eeg_values, cz_idx, pz_idx)
        keep_indices = [selected_channels.index('C3'), selected_channels.index('C4')]
        eeg_values = eeg_values[keep_indices, :]  # (2, n_samples)

        # Normalize EEG
        eeg_values = self._normalize(eeg_values)

        # Segment selection (from 4s to 7s)
        fs = 250
        start_sample = int(4 * fs)
        end_sample = int(7 * fs)
        eeg_values = eeg_values[:, start_sample:end_sample]  # (2, 750)
        eeg_tensor = torch.from_numpy(eeg_values.astype(np.float32)).transpose(0, 1)  # (750, 2)

        # Motion channels
        acc_channels = ['AccX', 'AccY', 'AccZ']
        gyro_channels = ['Gyro1', 'Gyro2', 'Gyro3']
        acc_values = eeg_data[acc_channels].values.astype(np.float32).T  # (3, n_samples)
        gyro_values = eeg_data[gyro_channels].values.astype(np.float32).T  # (3, n_samples)

        # Convert to physical units
        #acc_values = self._convert_acc(acc_values)
        gyro_values = self._convert_gyro(gyro_values)

        # Normalize motion data (optional, can comment out if not needed)
        #acc_values = self._normalize(acc_values)
        #gyro_values = self._normalize(gyro_values)

        # Segment selection for motion
        acc_values = acc_values[:, start_sample:end_sample]    # (3, 750)
        gyro_values = gyro_values[:, start_sample:end_sample]  # (3, 750)

        # Combine motion data
        motion_values = np.concatenate([acc_values, gyro_values], axis=0)  # (6, 750)
        motion_tensor = torch.from_numpy(motion_values.astype(np.float32)).transpose(0, 1)  # (750, 6)

        # Label
        if 'label' in row:
            label_tensor = torch.tensor(self.label_encoding[row['label']], dtype=torch.long)
        else:
            label_tensor = torch.tensor(-1, dtype=torch.long)  # Placeholder for test data

        return eeg_tensor, motion_tensor, label_tensor, row['id']       
        
        
class MotionAwareDenoiser(nn.Module):
    """ICA-inspired denoiser with motion artifact removal"""
    def __init__(self, eeg_channels=7, motion_channels=6, hidden_dim=64):
        super().__init__()
        
        # EEG processing branch
        self.eeg_projection = nn.Sequential(
            nn.Conv1d(eeg_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU()
        )
        
        # Motion processing branch
        self.motion_projection = nn.Sequential(
            nn.Conv1d(motion_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU()
        )
        
        # Temporal feature extraction
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim*2, hidden_dim*2, kernel_size=15, padding=7, groups=2),
            nn.BatchNorm1d(hidden_dim*2),
            nn.ELU(),
            nn.Conv1d(hidden_dim*2, hidden_dim*2, kernel_size=15, padding=7, groups=2),
            nn.BatchNorm1d(hidden_dim*2),
            nn.ELU()
        )
        
        # Attention mechanism to focus on motion artifacts
        self.attention = nn.Sequential(
            nn.Conv1d(hidden_dim*2, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Reconstruction to EEG space
        self.reconstruction = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Conv1d(hidden_dim, eeg_channels, kernel_size=1)
        )

    def forward(self, eeg, motion):
        # Project both signals to common space
        eeg_features = self.eeg_projection(eeg)
        motion_features = self.motion_projection(motion)
        
        # Concatenate features
        combined = torch.cat([eeg_features, motion_features], dim=1)
        
        # Extract temporal features
        temporal_features = self.temporal_conv(combined)
        
        # Attention mask for artifact regions
        attention_mask = self.attention(temporal_features)
        
        # Apply attention to EEG features
        attended_eeg = eeg_features * (1 - attention_mask)
        
        # Reconstruct clean EEG
        clean_eeg = self.reconstruction(attended_eeg)
        
        return clean_eeg

class SpatialAttentionClassifier(nn.Module):
    """Classifier with spatial attention"""
    def __init__(self, input_channels=7, num_classes=4, seq_length=500):
        super().__init__()
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(input_channels, input_channels//2, 1),
            nn.ELU(),
            nn.Conv1d(input_channels//2, input_channels, 1),
            nn.Sigmoid()
        )
        
        # Temporal feature extraction
        self.temporal_features = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=25, padding=12),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.MaxPool1d(4),
            
            nn.Conv1d(64, 128, kernel_size=15, padding=7),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.MaxPool1d(4),
            
            nn.Conv1d(128, 256, kernel_size=15, padding=7),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Apply spatial attention
        spatial_weights = self.spatial_attention(x)
        x_attended = x * spatial_weights
        
        # Extract temporal features
        features = self.temporal_features(x_attended)
        features = features.squeeze(-1)
        
        # Classify
        return self.classifier(features)

class EEGPipeline(nn.Module):
    """Simplified end-to-end pipeline"""
    def __init__(self, eeg_channels=7, motion_channels=6, num_classes=4):
        super().__init__()
        
        self.denoiser = MotionAwareDenoiser(eeg_channels, motion_channels)
        self.classifier = SpatialAttentionClassifier(eeg_channels, num_classes)
        
        # Normalization layers
        self.eeg_norm = nn.BatchNorm1d(eeg_channels)
        self.motion_norm = nn.BatchNorm1d(motion_channels)
        
    def forward(self, eeg, motion):
        # Normalize inputs
        eeg = self.eeg_norm(eeg)
        motion = self.motion_norm(motion)
        
        # Denoise EEG using motion information
        clean_eeg = self.denoiser(eeg, motion)
        
        # Classification
        logits = self.classifier(clean_eeg)
        
        return logits