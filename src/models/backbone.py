"""
3D-CNN Backbone for Hyperspectral Image Feature Extraction

Implements a lightweight 3D Convolutional Neural Network for extracting
spatial-spectral features from HSI patches.
"""

import torch
import torch.nn as nn
from typing import Tuple


class Simple3DCNN(nn.Module):
    """
    Simple 3D-CNN backbone for HSI feature extraction

    Architecture:
        Conv3D -> BN -> ReLU -> Conv3D -> BN -> ReLU -> ... -> Flatten -> FC

    Designed to be lightweight for 16GB VRAM constraint while effectively
    capturing spatial-spectral features.

    Args:
        input_channels: Number of input channels (usually 1 for HSI)
        spectral_size: Number of spectral bands (after PCA)
        spatial_size: Spatial dimension of input patches (e.g., 9 for 9x9)
        d_model: Output feature dimension

    Input shape: (B, 1, spectral_size, spatial_size, spatial_size)
    Output shape: (B, d_model)
    """

    def __init__(
        self,
        input_channels: int = 1,
        spectral_size: int = 30,
        spatial_size: int = 9,
        d_model: int = 128,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.spectral_size = spectral_size
        self.spatial_size = spatial_size
        self.d_model = d_model

        # Block 1: Focus on spectral features
        # Input: (B, 1, 30, 9, 9)
        self.conv1 = nn.Conv3d(
            in_channels=input_channels,
            out_channels=8,
            kernel_size=(7, 3, 3),
            stride=(2, 1, 1),
            padding=(0, 1, 1),
        )
        self.bn1 = nn.BatchNorm3d(8)
        self.relu1 = nn.ReLU(inplace=True)
        # Output: (B, 8, 12, 9, 9)

        # Block 2: Extract spatial-spectral features
        # Input: (B, 8, 12, 9, 9)
        self.conv2 = nn.Conv3d(
            in_channels=8,
            out_channels=16,
            kernel_size=(5, 3, 3),
            stride=(1, 1, 1),
            padding=(0, 1, 1),
        )
        self.bn2 = nn.BatchNorm3d(16)
        self.relu2 = nn.ReLU(inplace=True)
        # Output: (B, 16, 8, 9, 9)

        # Block 3: Deep feature extraction
        # Input: (B, 16, 8, 9, 9)
        self.conv3 = nn.Conv3d(
            in_channels=16,
            out_channels=32,
            kernel_size=(4, 3, 3),
            stride=(1, 1, 1),
            padding=(0, 1, 1),
        )
        self.bn3 = nn.BatchNorm3d(32)
        self.relu3 = nn.ReLU(inplace=True)
        # Output: (B, 32, 5, 9, 9)

        # Global Average Pooling to reduce spatial-spectral dimensions
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        # Output: (B, 32, 1, 1, 1)

        # Fully connected layer to project to d_model
        self.fc = nn.Linear(32, d_model)
        # Output: (B, d_model)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: (B, 1, spectral_size, spatial_size, spatial_size) tensor

        Returns:
            features: (B, d_model) tensor
        """
        # Ensure correct input shape
        if x.dim() == 4:
            # If input is (B, C, H, W), add spectral dimension
            # Assuming C is spectral bands, reshape to (B, 1, C, H, W)
            x = x.unsqueeze(1)

        # Block 1
        x = self.conv1(x)  # (B, 8, 12, 9, 9)
        x = self.bn1(x)
        x = self.relu1(x)

        # Block 2
        x = self.conv2(x)  # (B, 16, 8, 9, 9)
        x = self.bn2(x)
        x = self.relu2(x)

        # Block 3
        x = self.conv3(x)  # (B, 32, 5, 9, 9)
        x = self.bn3(x)
        x = self.relu3(x)

        # Global pooling
        x = self.gap(x)  # (B, 32, 1, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 32)

        # Final projection
        x = self.fc(x)  # (B, d_model)

        return x

    def get_output_dim(self) -> int:
        """Return the output feature dimension"""
        return self.d_model


class ResidualBlock3D(nn.Module):
    """
    3D Residual block for deeper architectures (optional enhancement)

    Can be used to build deeper networks while maintaining gradient flow
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int] = (3, 3, 3),
        stride: Tuple[int, int, int] = (1, 1, 1),
        padding: Tuple[int, int, int] = (1, 1, 1),
    ):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=(1, 1, 1),
            padding=padding,
        )
        self.bn2 = nn.BatchNorm3d(out_channels)

        # Shortcut connection
        if in_channels != out_channels or stride != (1, 1, 1):
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection"""
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the backbone
    model = Simple3DCNN(input_channels=1, spectral_size=30, spatial_size=9, d_model=128)

    # Test input
    batch_size = 4
    x = torch.randn(batch_size, 30, 9, 9)  # (B, C, H, W)

    # Forward pass
    features = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Number of parameters: {count_parameters(model):,}")

    # Expected output: (4, 128)
