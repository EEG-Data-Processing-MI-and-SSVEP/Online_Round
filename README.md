# EEG Classification Pipeline

End-to-end pipeline for Motor Imagery (MI) and Steady-State Visual Evoked Potential (SSVEP) classification using EEG and motion data.

## Table of Contents
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
  - [Motor Imagery Classification](#1-motor-imagery-mi-classification)
  - [SSVEP Classification](#2-ssvep-classification)
- [Model Architectures](#model-architectures)
  - [MI Model](#1-motor-imagery-pipeline-modelsmi_modelpy)
  - [SSVEP Model](#2-ssvep-model-modelsssvep_modelpy)
- [Data Processing](#data-processing)
- [Training Pipeline](#training-pipeline)
- [Evaluation Metrics](#evaluation-metrics)
- [Reproducibility](#reproducibility)
- [License](#license)

## Repository Structure
```
├── checkpoints/ # Pre-trained model checkpoints
│ ├── mi_model.pth # Motor Imagery model
│ └── ssvep_model.pth # SSVEP model
│
├── configs/ # Configuration files
│ ├── mi_config.yaml # MI training/inference config
│ └── ssvep_config.yaml # SSVEP training/inference config
│
├── data/ # Data directory
│ ├── raw/ # Raw data files
│ └── processed/ # Processed data
│
├── models/ # Model architectures
│ ├── mi_model.py # MI classification model
│ └── ssvep_model.py # SSVEP classification model
│
├── scripts/ # Utility scripts
│ ├── preprocessing/ # Data preprocessing
│ ├── training/ # Training scripts
│ └── evaluation/ # Evaluation metrics
│
├── requirements.txt # Python dependencies
├── train.py # Main training script
├── predict.py # Main inference script
└── README.md # This file
```
## Requirements
For installing requirements
```bash
pip install -r requirements.txt
```

## Quick Start
### 1. Motor Imagery (MI) Classification

#### Training:
```bash
python train.py \
    --task MI \
    --config configs/mi_config.yaml \
    --data_dir data/raw \
    --output_dir checkpoints/
```

#### Inference:
```bash
python predict.py \
    --task MI \
    --checkpoint checkpoints/mi_model.pt \
    --input data/test_samples.csv \
    --output predictions.json
```

### 2. SSVEP Classification

#### Training:
```bash
python train.py \
    --task SSVEP \
    --config configs/ssvep_config.yaml \
    --data_dir data/raw \
    --output_dir checkpoints/
```
#### Inference:
```bash
python predict.py \
    --task SSVEP \
    --checkpoint checkpoints/ssvep_model.pt \
    --input data/test_samples.csv \
    --output predictions.json
```

## Model Architectures
### 1. Motor Imagery Pipeline (`models/mi_model.py`)

#### Components:
```python
class MotionAwareDenoiser(nn.Module):
    """ICA-inspired denoiser with motion artifact removal"""
    # Architecture details...
    # - EEG and motion processing branches
    # - Temporal convolution layers
    # - Attention mechanism
    # - Reconstruction layer

class SpatialAttentionClassifier(nn.Module):
    """Classifier with spatial attention"""
    # Architecture details...
    # - Spatial attention layer
    # - Temporal feature extraction (3 conv blocks)
    # - Classification head

class EEGPipeline(nn.Module):
    """End-to-end MI classification pipeline"""
    # Combines denoiser and classifier
```

### 2. SSVEP Model (`models/ssvep_model.py`)

#### Components:
```python
class EnhancedSSVEPModel(nn.Module):
    """Enhanced model with frequency attention"""
    # Architecture details...
    # - EEG branch with residual blocks
    # - Frequency branch for SSVEP features
    # - Motion branch for artifact handling
    # - Attention pooling
    # - Classifier head

class ResidualBlock(nn.Module):
    """Custom 1D residual block"""
    # Implements skip connections

class AttentionPooling(nn.Module):
    """Learnable temporal pooling"""
    # Implements attention-based pooling
```

## Data Processing Pipeline:
### SSVEP Data Processing

#### Processing Pipeline

1. **EEG Processing**:
   - **Channel Selection**: PO7, OZ, PO8
   - **Notch Filtering**: 50Hz removal
   - **Bandpass Filtering**: 6-30Hz range
   - **Referencing**: Average reference (`set_eeg_reference('average')`)
   - **Normalization**: Channel-wise z-score
   - **Frequency Feature Extraction**:
     - Welch's PSD estimation
     - Power extraction in SSVEP bands:
       - Left: 9-11Hz (10Hz target)
       - Right: 12-14Hz (13Hz target)
       - Forward: 6-8Hz (7Hz target)
       - Backward: 7-9Hz (8Hz target)

2. **Motion Data Processing**:
   - **Channels**: AccX/Y/Z, Gyro1/2/3
   - **Normalization**: Channel-wise z-score
   - **Unit Conversion**: Gyroscope values converted to deg/s

3. **Temporal Processing**:
   - Full trial length: 1750 samples (7s at 250Hz)
   - Output shape: (3 channels × 1750 samples)

#### Implementation (`SSVEPDataset` class)

```python
class SSVEPDataset(Dataset):
    FREQUENCIES = {
        'Left': 10,     # 10Hz stimulation
        'Right': 13,    # 13Hz stimulation  
        'Forward': 7,   # 7Hz stimulation
        'Backward': 8   # 8Hz stimulation
    }
    
    def __getitem__(self, idx):
        # Load and preprocess data:
        # 1. Apply 50Hz notch filter
        # 2. Apply 6-30Hz bandpass
        # 3. Average reference
        # 4. Extract frequency features
        # 5. Normalize all channels
        
        return (
            eeg_tensor,      # (3, 1750) filtered EEG
            freq_tensor,     # (3, 4) frequency features  
            motion_tensor,   # (6, 1750) motion data
            label_tensor     # class label
        )
```
###  Motor Imagery (MI) Data Processing
#### Processing Pipeline

1. **EEG Processing:**

    - **Channel Selection:** C3, C4 (after CAR)
    - **Bandpass Filtering:** 8-30Hz range
    - **Referencing:** Common Average Reference using CZ and PZ
    - **Normalization:** Channel-wise z-score
    - **Temporal Window:** 2-6s (1000 samples)

2. **Motion Data Processing:**

    - **Channels:** AccX/Y/Z, Gyro1/2/3
    - **Unit Conversion:** Gyroscope to deg/s
    - **Normalization:** Channel-wise z-score (optional)
    - **Temporal Window:** 2-6s (1000 samples)


### Implementation (EEGDataset class)
```python
class EEGDataset(Dataset):
    label_encoding = {'Left': 0, 'Right': 1}
    
    def __getitem__(self, idx):
        # Processing steps:
        # 1. Load C3, CZ, C4, PZ channels
        # 2. Apply 8-30Hz bandpass
        # 3. CAR reference using CZ/PZ
        # 4. Keep only C3/C4
        # 5. Select 2-6s window
        # 6. Normalize
        
        return (
            eeg_tensor,      # (1000, 2) [C3, C4]
            motion_tensor,   # (1000, 6) motion data  
            label_tensor     # class label (0=Left, 1=Right)
        )
```

### Key Differences Between Modalities
Feature	SSVEP	Motor Imagery
Channels	PO7, OZ, PO8	C3, C4
Filtering	6-30Hz + 50Hz notch	8-30Hz only
Reference	Average reference	CAR (CZ/PZ)
Window	Full trial (1750 samples)	2-6s (1000 samples)
Features	Frequency band power	Raw time-series
Classes	4 directions	2 classes (Left/Right)
Common Processing Elements

    Normalization: Both use channel-wise z-score normalization

    Motion Data: Both process accelerometer and gyroscope data

    Sampling Rate: 250Hz for both modalities

    PyTorch Integration: Both return ready-to-use tensors

Usage Example
python

# SSVEP Dataset
ssvep_dataset = SSVEPDataset(csv_path="metadata.csv")
ssvep_loader = DataLoader(ssvep_dataset, batch_size=32)

# MI Dataset
mi_dataset = EEGDataset(csv_path="metadata.csv", task="MI")
mi_loader = DataLoader(mi_dataset, batch_size=32)

This documentation reflects the exact processing pipeline implemented in your code, with clear specifications of all parameters and processing steps for both SSVEP and Motor Imagery classification tasks.