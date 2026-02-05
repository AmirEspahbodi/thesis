"""Data loading and preprocessing utilities for hyperspectral images.

This module handles loading HSI data from .mat files, applying PCA for
spectral reduction, normalization, and 3D patch extraction.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class HSIDataLoader:
    """Loader for hyperspectral image data from .mat files.

    This class handles loading, preprocessing, and patch extraction
    from HSI datasets stored in MATLAB format.
    """

    def __init__(
        self,
        data_path: str,
        gt_path: str,
        hsi_key: str = "indian_pines_corrected",
        gt_key: str = "indian_pines_gt",
    ):
        """Initialize the HSI data loader.

        Args:
            data_path: Path to the .mat file.
            hsi_key: Key for the HSI cube in the .mat file.
            gt_key: Key for the ground truth labels in the .mat file.

        Raises:
            FileNotFoundError: If the data file doesn't exist.
            KeyError: If the specified keys are not in the .mat file.
        """
        self.data_path = Path(data_path)
        self.gt_path = Path(gt_path)
        self.hsi_key = hsi_key
        self.gt_key = gt_key

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        if not self.gt_path.exists():
            raise FileNotFoundError(f"GT file not found: {data_path}")

        self.hsi_data: Optional[np.ndarray] = None
        self.gt_data: Optional[np.ndarray] = None
        self.preprocessed_data: Optional[np.ndarray] = None

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load HSI cube and ground truth from .mat file.

        Returns:
            Tuple of (hsi_data, gt_data) as numpy arrays.

        Raises:
            KeyError: If specified keys not found in .mat file.
        """
        try:
            hsi_data = sio.loadmat(str(self.data_path))
            gt_data = sio.loadmat(str(self.gt_path))

            if self.hsi_key not in hsi_data:
                raise KeyError(f"HSI key '{self.hsi_key}' not found in .mat file")
            if self.gt_key not in gt_data:
                raise KeyError(f"GT key '{self.gt_key}' not found in .mat file")

            self.hsi_data = hsi_data[self.hsi_key].astype(np.float32)
            self.gt_data = gt_data[self.gt_key].astype(np.int32)

            print(f"Loaded HSI data with shape: {self.hsi_data.shape}")
            print(f"Loaded GT data with shape: {self.gt_data.shape}")
            print(
                f"Number of classes: {len(np.unique(self.gt_data)) - 1}"
            )  # -1 for background

            return self.hsi_data, self.gt_data

        except Exception as e:
            raise RuntimeError(f"Error loading data from {self.data_path}: {str(e)}")

    def apply_pca(
        self, n_components: int = 30, data: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply PCA for spectral dimension reduction.

        Args:
            n_components: Number of principal components to retain.
            data: HSI data to transform. If None, uses self.hsi_data.

        Returns:
            Transformed HSI data with reduced spectral dimension.

        Raises:
            ValueError: If data is None and self.hsi_data is not loaded.
        """
        if data is None:
            if self.hsi_data is None:
                raise ValueError("No HSI data loaded. Call load_data() first.")
            data = self.hsi_data

        height, width, bands = data.shape

        # Reshape to (n_samples, n_features) for PCA
        reshaped = data.reshape(-1, bands)

        # Apply PCA
        pca = PCA(n_components=n_components, random_state=42)
        transformed = pca.fit_transform(reshaped)

        # Reshape back to (height, width, n_components)
        result = transformed.reshape(height, width, n_components)

        explained_variance = np.sum(pca.explained_variance_ratio_)
        print(f"PCA: Reduced {bands} bands to {n_components} components")
        print(f"Explained variance: {explained_variance:.4f}")

        return result.astype(np.float32)

    def normalize_data(self, data: np.ndarray, method: str = "minmax") -> np.ndarray:
        """Normalize the HSI data.

        Args:
            data: HSI data to normalize.
            method: Normalization method ('minmax' or 'zscore').

        Returns:
            Normalized HSI data.

        Raises:
            ValueError: If method is not supported.
        """
        height, width, bands = data.shape
        reshaped = data.reshape(-1, bands)

        if method == "minmax":
            scaler = MinMaxScaler(feature_range=(0, 1))
        elif method == "zscore":
            scaler = StandardScaler()
        else:
            raise ValueError(f"Unsupported normalization method: {method}")

        normalized = scaler.fit_transform(reshaped)
        result = normalized.reshape(height, width, bands)

        print(f"Applied {method} normalization")

        return result.astype(np.float32)

    def preprocess(
        self, n_components: int = 30, normalization: str = "minmax"
    ) -> np.ndarray:
        """Complete preprocessing pipeline: PCA + normalization.

        Args:
            n_components: Number of PCA components.
            normalization: Normalization method.

        Returns:
            Preprocessed HSI data.
        """
        if self.hsi_data is None:
            self.load_data()

        # Apply PCA
        pca_data = self.apply_pca(n_components=n_components)

        # Apply normalization
        self.preprocessed_data = self.normalize_data(pca_data, method=normalization)

        return self.preprocessed_data


class PatchExtractor:
    """Extract 3D spatial-spectral patches from HSI data.

    This class handles extraction of cubic patches around each pixel,
    with proper boundary padding to handle edge cases.
    """

    def __init__(self, patch_size: int = 9, padding_mode: str = "reflect"):
        """Initialize the patch extractor.

        Args:
            patch_size: Spatial size of patches (creates patch_size x patch_size).
            padding_mode: Padding mode for boundaries ('reflect', 'constant', 'edge').
        """
        self.patch_size = patch_size
        self.padding_mode = padding_mode
        self.pad_width = patch_size // 2

    def extract_patches(
        self, hsi_data: np.ndarray, gt_data: np.ndarray, remove_background: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract 3D patches for all labeled pixels.

        Args:
            hsi_data: Preprocessed HSI data (H, W, B).
            gt_data: Ground truth labels (H, W).
            remove_background: Whether to exclude background pixels (label=0).

        Returns:
            Tuple of (patches, labels):
                - patches: Array of shape (N, patch_size, patch_size, B)
                - labels: Array of shape (N,)
        """
        height, width, bands = hsi_data.shape

        # Pad the HSI data to handle boundaries
        padded_hsi = np.pad(
            hsi_data,
            (
                (self.pad_width, self.pad_width),
                (self.pad_width, self.pad_width),
                (0, 0),
            ),
            mode=self.padding_mode,
        )

        # Get coordinates of labeled pixels
        if remove_background:
            coords = np.argwhere(gt_data > 0)
        else:
            coords = np.argwhere(gt_data >= 0)

        num_samples = len(coords)
        patches = np.zeros(
            (num_samples, self.patch_size, self.patch_size, bands), dtype=np.float32
        )
        labels = np.zeros(num_samples, dtype=np.int64)

        # Extract patches
        for idx, (i, j) in enumerate(coords):
            # Adjust indices for padding
            pi, pj = i + self.pad_width, j + self.pad_width

            # Extract patch
            patch = padded_hsi[
                pi - self.pad_width : pi + self.pad_width + 1,
                pj - self.pad_width : pj + self.pad_width + 1,
                :,
            ]

            patches[idx] = patch
            labels[idx] = gt_data[i, j]

        print(
            f"Extracted {num_samples} patches of size {self.patch_size}x{self.patch_size}x{bands}"
        )

        return patches, labels

    def extract_single_patch(
        self, hsi_data: np.ndarray, row: int, col: int
    ) -> np.ndarray:
        """Extract a single patch at given coordinates.

        Args:
            hsi_data: HSI data (H, W, B).
            row: Row coordinate.
            col: Column coordinate.

        Returns:
            Extracted patch of shape (patch_size, patch_size, B).
        """
        height, width, bands = hsi_data.shape

        # Pad data
        padded_hsi = np.pad(
            hsi_data,
            (
                (self.pad_width, self.pad_width),
                (self.pad_width, self.pad_width),
                (0, 0),
            ),
            mode=self.padding_mode,
        )

        # Adjust indices for padding
        pi, pj = row + self.pad_width, col + self.pad_width

        # Extract patch
        patch = padded_hsi[
            pi - self.pad_width : pi + self.pad_width + 1,
            pj - self.pad_width : pj + self.pad_width + 1,
            :,
        ]

        return patch.astype(np.float32)
