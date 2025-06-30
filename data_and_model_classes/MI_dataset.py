import os, sys, torch
from dotenv import load_dotenv
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# Get the directory of the current file
current_file_dir = os.path.dirname(__file__)

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(current_file_dir, '..'))

# Add the parent directory to Python path
sys.path.append(parent_dir)

# Get the base directory path for being able to read .env file which is in the parent directory
base_directory_path = os.getenv('BASE_DIR_PATH')
load_dotenv()

class EEGDataset(Dataset):
    
    base_path = os.getenv('BASE_DIR_PATH')
    label_encoding = {'Left': 0, 'Right': 1}

    # Sensitivity values (replace with your device's actual values if different)
    gyro_scale  = 0.02  # deg/s per LSB
    gyro_offset = 0
    acc_scale   = 0.000598  # g per LSB
    acc_offset  = 0

    def __init__(self, csv_path, task=None, segment_length=250, overlap=0.5):
        self.metadata = pd.read_csv(csv_path)
        self.task = task

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

    def _bandpass_filter(self, data, lowcut=8, highcut=30, fs=250, order=5):
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

        # Segment selection (from 2nd to 6th)
        fs = 250
        start_sample = int(2 * fs)
        end_sample = int(6 * fs)
        eeg_values = eeg_values[:, start_sample:end_sample]  # (2, 1000)
        eeg_tensor = torch.from_numpy(eeg_values.astype(np.float32)).transpose(0, 1)  # (1000, 2)

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

        label_tensor = torch.tensor(self.label_encoding[row['label']], dtype=torch.long)

        return eeg_tensor, motion_tensor, label_tensor