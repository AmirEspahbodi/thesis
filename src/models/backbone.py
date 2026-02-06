"""3D-CNN backbone for hyperspectral image feature extraction.

This module implements a Residual 3D convolutional neural network with
Squeeze-and-Excitation (SE) attention, optimized for extracting
spatial-spectral features from HSI patches.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation Block for 3D data.

    Performs channel-wise recalibration of feature maps.
    Structure: Global Avg Pool -> Reduce -> ReLU -> Expand -> Sigmoid -> Scale.
    """

    def __init__(self, channels: int, reduction: int = 16):
        """Initialize the SE Block.

        Args:
            channels: Number of input channels.
            reduction: Reduction ratio for the bottleneck in excitation.
        """
        super(SEBlock3D, self).__init__()

        # Ensure hidden channels is at least 1
        mid_channels = max(1, channels // reduction)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SE block.

        Args:
            x: Input tensor of shape (B, C, D, H, W).

        Returns:
            Recalibrated tensor of same shape.
        """
        b, c, _, _, _ = x.size()

        # Squeeze: Global Average Pooling -> (B, C, 1, 1, 1) -> (B, C)
        y = self.avg_pool(x).view(b, c)

        # Excitation: FC -> ReLU -> FC -> Sigmoid -> (B, C) -> (B, C, 1, 1, 1)
        y = self.fc(y).view(b, c, 1, 1, 1)

        # Scale: Element-wise multiplication
        return x * y.expand_as(x)


class ResidualSEBlock(nn.Module):
    """Residual 3D Block with SE Attention.

    Structure:
    Input -> [Conv3D -> BN -> SE] + [Skip Connection] -> ReLU -> MaxPool -> Dropout
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int] = (3, 3, 3),
        pool_size: Tuple[int, int, int] = (2, 2, 2),
        dropout_rate: float = 0.3,
        reduction: int = 16,
    ):
        """Initialize the Residual SE Block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolutional kernel.
            pool_size: Size of the pooling window.
            dropout_rate: Dropout probability.
            reduction: SE block reduction ratio.
        """
        super(ResidualSEBlock, self).__init__()

        # Calculate padding to maintain spatial dimensions before pooling
        # Padding = kernel_size // 2 assures 'same' padding for odd kernels
        padding = tuple(k // 2 for k in kernel_size)

        # Main convolutional branch
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm3d(out_channels)

        # Squeeze-and-Excitation Attention
        self.se = SEBlock3D(out_channels, reduction=reduction)

        # Shortcut/Skip connection branch
        # If dimensions change (channel count), use 1x1x1 convolution to match
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        # Activation and regularization
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(kernel_size=pool_size, stride=pool_size)
        self.dropout = nn.Dropout3d(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the residual block.

        Args:
            x: Input tensor of shape (B, C, D, H, W).

        Returns:
            Output tensor.
        """
        # Main branch
        out = self.conv(x)
        out = self.bn(out)
        out = self.se(out)  # Apply attention before addition

        # Residual connection
        residual = self.shortcut(x)
        out += residual

        # Activation
        out = self.relu(out)

        # Pooling and Dropout (performed after the residual block logic)
        out = self.pool(out)
        out = self.dropout(out)

        return out


class HSI3DCNN(nn.Module):
    """Residual 3D-CNN backbone for HSI feature extraction.

    This network processes 3D spatial-spectral patches using residual blocks
    with attention mechanisms and produces fixed-dimensional embedding vectors.
    """

    def __init__(
        self,
        input_channels: int = 1,
        spectral_depth: int = 30,
        spatial_size: int = 9,
        conv_channels: Tuple[int, ...] = (32, 64, 128),
        kernel_sizes: Tuple[Tuple[int, int, int], ...] = (
            (3, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
        ),
        pool_sizes: Tuple[Tuple[int, int, int], ...] = (
            (2, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
        ),
        embedding_dim: int = 256,
        dropout_rate: float = 0.3,
    ):
        """Initialize the Residual 3D-CNN backbone.

        Args:
            input_channels: Number of input channels (typically 1 for HSI).
            spectral_depth: Depth of spectral dimension after PCA.
            spatial_size: Spatial size of input patches (e.g., 9 for 9x9).
            conv_channels: Tuple of output channels for each conv layer.
            kernel_sizes: Tuple of kernel sizes for each conv layer.
            pool_sizes: Tuple of pooling sizes for each layer.
            embedding_dim: Dimension of final embedding vector.
            dropout_rate: Dropout rate for regularization.
        """
        super(HSI3DCNN, self).__init__()

        self.input_channels = input_channels
        self.spectral_depth = spectral_depth
        self.spatial_size = spatial_size
        self.embedding_dim = embedding_dim

        # Build residual layers
        layers = []
        in_ch = input_channels

        for i, out_ch in enumerate(conv_channels):
            # Safe indexing for kernels and pools in case lists are shorter than channels
            kernel_size = kernel_sizes[i] if i < len(kernel_sizes) else (3, 3, 3)
            pool_size = pool_sizes[i] if i < len(pool_sizes) else (2, 2, 2)

            layers.append(
                ResidualSEBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    pool_size=pool_size,
                    dropout_rate=dropout_rate,
                    reduction=16,
                )
            )
            in_ch = out_ch

        self.conv_layers = nn.Sequential(*layers)

        # Calculate the size after all convolutions and pooling
        self.feature_size = self._calculate_feature_size()

        # Fully connected layers for embedding
        self.fc_layers = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, embedding_dim),
        )

        # Initialize weights
        self._initialize_weights()

    def _calculate_feature_size(self) -> int:
        """Calculate the flattened feature size after conv layers.

        Returns:
            Size of flattened features.
        """
        # Create a dummy input
        dummy_input = torch.zeros(
            1,
            self.input_channels,
            self.spectral_depth,
            self.spatial_size,
            self.spatial_size,
        )

        # Pass through conv layers
        with torch.no_grad():
            dummy_output = self.conv_layers(dummy_input)

        # Return flattened size
        return int(torch.prod(torch.tensor(dummy_output.shape[1:])))

    def _initialize_weights(self):
        """Initialize network weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (B, 1, D, H, W).

        Returns:
            Embedding tensor of shape (B, embedding_dim).
        """
        # Pass through residual convolutional layers
        x = self.conv_layers(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Pass through FC layers
        x = self.fc_layers(x)

        # L2 Normalization for metric learning stability
        x = F.normalize(x, p=2, dim=1)

        return x

    def get_embedding_dim(self) -> int:
        """Get the embedding dimension.

        Returns:
            Embedding dimension.
        """
        return self.embedding_dim
