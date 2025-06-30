import torch
import torch.nn as nn

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
