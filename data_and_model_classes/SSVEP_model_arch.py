import torch
import torch.nn as nn

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