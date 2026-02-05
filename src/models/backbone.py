"""3D-CNN backbone for hyperspectral image feature extraction.

This module implements a 3D convolutional neural network optimized for
extracting spatial-spectral features from HSI patches.
"""

from typing import List, Tuple

import torch
import torch.nn as nn


class Conv3DBlock(nn.Module):
    """3D Convolutional block with BatchNorm, ReLU, and Pooling.

    This building block is used to construct the 3D-CNN backbone.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int] = (3, 3, 3),
        pool_size: Tuple[int, int, int] = (2, 2, 2),
        dropout_rate: float = 0.3,
    ):
        """Initialize the 3D conv block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolutional kernel.
            pool_size: Size of the pooling window.
            dropout_rate: Dropout probability.
        """
        super(Conv3DBlock, self).__init__()

        # Calculate padding to maintain spatial dimensions before pooling
        padding = tuple(k // 2 for k in kernel_size)

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(kernel_size=pool_size, stride=pool_size)
        self.dropout = nn.Dropout3d(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the conv block.

        Args:
            x: Input tensor of shape (B, C, D, H, W).

        Returns:
            Output tensor after convolution, normalization, activation, and pooling.
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class HSI3DCNN(nn.Module):
    """3D-CNN backbone for HSI feature extraction.

    This network processes 3D spatial-spectral patches and produces
    fixed-dimensional embedding vectors.
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
        """Initialize the 3D-CNN backbone.

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

        # Build convolutional layers
        layers = []
        in_ch = input_channels

        for i, out_ch in enumerate(conv_channels):
            kernel_size = kernel_sizes[i] if i < len(kernel_sizes) else (3, 3, 3)
            pool_size = pool_sizes[i] if i < len(pool_sizes) else (2, 2, 2)

            layers.append(
                Conv3DBlock(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    pool_size=pool_size,
                    dropout_rate=dropout_rate,
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
        # Pass through convolutional layers
        x = self.conv_layers(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Pass through FC layers
        x = self.fc_layers(x)

        x = torch.nn.functional.normalize(x, p=2, dim=1)

        return x

    def get_embedding_dim(self) -> int:
        """Get the embedding dimension.

        Returns:
            Embedding dimension.
        """
        return self.embedding_dim
