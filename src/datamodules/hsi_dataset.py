"""
Hyperspectral Image Dataset Module

This module handles loading, preprocessing, and patch extraction from HSI .mat files.
Key features:
- PCA for spectral dimension reduction
- Spatial padding for boundary patches
- Efficient patch extraction with caching
"""

import os
from typing import Tuple, List, Optional, Dict
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class HSIDataset(Dataset):
    """
    Hyperspectral Image Dataset for Few-Shot Learning

    Handles:
    1. Loading .mat files
    2. PCA-based spectral reduction (n_bands -> target_bands)
    3. Spatial padding for edge patches
    4. Patch extraction centered at labeled pixels

    Args:
        data_root: Root directory containing .mat files
        file_name: Name of the HSI data file (e.g., 'Houston13.mat')
        gt_name: Name of the ground truth file
        image_key: Key to access image data in .mat file
        gt_key: Key to access ground truth in .mat file
        patch_size: Spatial size of extracted patches (e.g., 9 for 9x9)
        target_bands: Target number of spectral bands after PCA
        ignored_labels: List of labels to ignore (usually [0] for background)
        indices: Optional list of specific indices to use (for train/val/test split)
    """

    def __init__(
        self,
        data_root: str,
        file_name: str,
        gt_name: str,
        image_key: str,
        gt_key: str,
        patch_size: int = 9,
        target_bands: int = 30,
        ignored_labels: List[int] = [0],
        indices: Optional[List[int]] = None,
    ):
        super().__init__()

        self.data_root = data_root
        self.patch_size = patch_size
        self.target_bands = target_bands
        self.ignored_labels = ignored_labels
        self.padding = patch_size // 2

        # Load data
        self.image, self.gt = self._load_data(file_name, gt_name, image_key, gt_key)

        # Apply PCA for spectral reduction
        self.image = self._apply_pca(self.image, target_bands)

        # Pad image for boundary patches
        self.image = self._pad_image(self.image)

        # Get valid sample indices (non-background, labeled pixels)
        self.valid_indices = self._get_valid_indices()

        # Use provided indices if specified, otherwise use all valid indices
        if indices is not None:
            self.indices = indices
        else:
            self.indices = self.valid_indices

        # Create label mapping (ignore background/unlabeled)
        self.label_map = self._create_label_map()

    def _load_data(
        self, file_name: str, gt_name: str, image_key: str, gt_key: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load HSI image and ground truth from .mat files

        Returns:
            image: (H, W, C) numpy array
            gt: (H, W) numpy array with class labels
        """
        image_path = os.path.join(self.data_root, file_name)
        gt_path = os.path.join(self.data_root, gt_name)

        # Load image data
        image_mat = loadmat(image_path)
        image = image_mat[image_key].astype(np.float32)

        # Load ground truth
        gt_mat = loadmat(gt_path)
        gt = gt_mat[gt_key].astype(np.int64)

        # Ensure 2D ground truth
        if gt.ndim == 3:
            gt = gt.squeeze()

        return image, gt

    def _apply_pca(self, image: np.ndarray, n_components: int) -> np.ndarray:
        """
        Apply PCA to reduce spectral dimensions

        Args:
            image: (H, W, C) array
            n_components: Target number of spectral bands

        Returns:
            reduced_image: (H, W, n_components) array
        """
        H, W, C = image.shape

        # Skip PCA if already at target dimensions
        if C == n_components:
            return image

        # Reshape to (H*W, C) for PCA
        image_2d = image.reshape(-1, C)

        # Standardize before PCA
        scaler = StandardScaler()
        image_2d = scaler.fit_transform(image_2d)

        # Apply PCA
        pca = PCA(n_components=n_components)
        image_pca = pca.fit_transform(image_2d)

        # Reshape back to (H, W, n_components)
        image_pca = image_pca.reshape(H, W, n_components)

        print(
            f"PCA: Reduced {C} bands to {n_components} bands "
            f"(explained variance: {pca.explained_variance_ratio_.sum():.4f})"
        )

        return image_pca.astype(np.float32)

    def _pad_image(self, image: np.ndarray) -> np.ndarray:
        """
        Pad image spatially to handle boundary patches

        Args:
            image: (H, W, C) array

        Returns:
            padded_image: (H+2*padding, W+2*padding, C) array
        """
        pad_width = ((self.padding, self.padding), (self.padding, self.padding), (0, 0))

        # Use edge padding (replicate border pixels)
        padded = np.pad(image, pad_width, mode="edge")

        return padded

    def _get_valid_indices(self) -> List[int]:
        """
        Get indices of all valid (labeled, non-background) pixels

        Returns:
            List of linear indices into flattened ground truth
        """
        valid_mask = np.ones_like(self.gt, dtype=bool)

        # Exclude ignored labels
        for label in self.ignored_labels:
            valid_mask &= self.gt != label

        # Get linear indices
        valid_indices = np.where(valid_mask.ravel())[0].tolist()

        return valid_indices

    def _create_label_map(self) -> Dict[int, int]:
        """
        Create mapping from original labels to contiguous class indices

        Returns:
            Dictionary mapping original_label -> class_index (0-based)
        """
        unique_labels = np.unique(self.gt)
        valid_labels = [l for l in unique_labels if l not in self.ignored_labels]
        valid_labels = sorted(valid_labels)

        label_map = {label: idx for idx, label in enumerate(valid_labels)}

        return label_map

    def _extract_patch(self, x: int, y: int) -> torch.Tensor:
        """
        Extract a patch centered at (x, y) from the padded image

        Args:
            x, y: Center coordinates in the ORIGINAL (unpadded) image

        Returns:
            patch: (C, patch_size, patch_size) tensor
        """
        # Adjust coordinates for padding
        x_padded = x + self.padding
        y_padded = y + self.padding

        # Extract patch
        patch = self.image[
            x_padded - self.padding : x_padded + self.padding + 1,
            y_padded - self.padding : y_padded + self.padding + 1,
            :,
        ]

        # Convert to tensor and rearrange to (C, H, W)
        patch = torch.from_numpy(patch).permute(2, 0, 1)

        return patch

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample (patch, label) pair

        Args:
            idx: Index in self.indices

        Returns:
            patch: (C, patch_size, patch_size) tensor
            label: Class index (0-based, after label mapping)
        """
        # Get linear index
        linear_idx = self.indices[idx]

        # Convert to 2D coordinates
        H, W = self.gt.shape
        x = linear_idx // W
        y = linear_idx % W

        # Get original label
        original_label = self.gt[x, y]

        # Map to class index
        label = self.label_map[original_label]

        # Extract patch
        patch = self._extract_patch(x, y)

        return patch, label

    def __len__(self) -> int:
        """Return number of available samples"""
        return len(self.indices)

    def get_num_classes(self) -> int:
        """Return number of classes (excluding background)"""
        return len(self.label_map)

    def get_class_counts(self) -> Dict[int, int]:
        """
        Get number of samples per class

        Returns:
            Dictionary mapping class_index -> count
        """
        counts = {}
        for idx in self.indices:
            H, W = self.gt.shape
            x = idx // W
            y = idx % W
            original_label = self.gt[x, y]
            class_idx = self.label_map[original_label]
            counts[class_idx] = counts.get(class_idx, 0) + 1

        return counts


def create_data_splits(
    dataset: HSIDataset,
    train_ratio: float = 0.1,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[HSIDataset, HSIDataset, HSIDataset]:
    """
    Split dataset into train/val/test sets for in-domain learning

    Splits are performed per-class to maintain class balance

    Args:
        dataset: The full dataset to split
        train_ratio: Fraction of samples for training
        val_ratio: Fraction of samples for validation
        seed: Random seed for reproducibility

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    np.random.seed(seed)

    # Group indices by class
    class_indices = {}
    for idx in dataset.valid_indices:
        H, W = dataset.gt.shape
        x = idx // W
        y = idx % W
        original_label = dataset.gt[x, y]
        class_idx = dataset.label_map[original_label]

        if class_idx not in class_indices:
            class_indices[class_idx] = []
        class_indices[class_idx].append(idx)

    train_indices = []
    val_indices = []
    test_indices = []

    # Split each class separately
    for class_idx, indices in class_indices.items():
        indices = np.array(indices)
        np.random.shuffle(indices)

        n_samples = len(indices)
        n_train = max(1, int(n_samples * train_ratio))
        n_val = max(1, int(n_samples * val_ratio))

        train_indices.extend(indices[:n_train].tolist())
        val_indices.extend(indices[n_train : n_train + n_val].tolist())
        test_indices.extend(indices[n_train + n_val :].tolist())

    # Create dataset objects with specific indices
    train_dataset = HSIDataset(
        data_root=dataset.data_root,
        file_name=None,  # Already loaded
        gt_name=None,
        image_key=None,
        gt_key=None,
        patch_size=dataset.patch_size,
        target_bands=dataset.target_bands,
        ignored_labels=dataset.ignored_labels,
        indices=train_indices,
    )
    # Copy loaded data to avoid reloading
    train_dataset.image = dataset.image
    train_dataset.gt = dataset.gt
    train_dataset.label_map = dataset.label_map
    train_dataset.valid_indices = dataset.valid_indices

    val_dataset = HSIDataset(
        data_root=dataset.data_root,
        file_name=None,
        gt_name=None,
        image_key=None,
        gt_key=None,
        patch_size=dataset.patch_size,
        target_bands=dataset.target_bands,
        ignored_labels=dataset.ignored_labels,
        indices=val_indices,
    )
    val_dataset.image = dataset.image
    val_dataset.gt = dataset.gt
    val_dataset.label_map = dataset.label_map
    val_dataset.valid_indices = dataset.valid_indices

    test_dataset = HSIDataset(
        data_root=dataset.data_root,
        file_name=None,
        gt_name=None,
        image_key=None,
        gt_key=None,
        patch_size=dataset.patch_size,
        target_bands=dataset.target_bands,
        ignored_labels=dataset.ignored_labels,
        indices=test_indices,
    )
    test_dataset.image = dataset.image
    test_dataset.gt = dataset.gt
    test_dataset.label_map = dataset.label_map
    test_dataset.valid_indices = dataset.valid_indices

    print(
        f"Data split - Train: {len(train_indices)}, "
        f"Val: {len(val_indices)}, Test: {len(test_indices)}"
    )

    return train_dataset, val_dataset, test_dataset
