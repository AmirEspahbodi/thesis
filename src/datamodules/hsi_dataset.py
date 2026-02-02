import os
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class HSIDataset(Dataset):
    """
    Hyperspectral Image Dataset for Few-Shot Learning
    ... (Keep existing class definition unchanged) ...
    """

    def __init__(
        self,
        data_root: str,
        file_name: Optional[str],
        gt_name: Optional[str],
        image_key: Optional[str],
        gt_key: Optional[str],
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

        # If file_name is provided, load and process data
        if file_name is not None:
            # Load data
            self.image, self.gt = self._load_data(file_name, gt_name, image_key, gt_key)

            # Apply PCA for spectral reduction
            self.image = self._apply_pca(self.image, target_bands)

            # Pad image for boundary patches
            self.image = self._pad_image(self.image)

            # Get valid sample indices (non-background, labeled pixels)
            self.valid_indices = self._get_valid_indices()

            # Create label mapping (ignore background/unlabeled)
            self.label_map = self._create_label_map()
        else:
            # Placeholder initialization
            self.image = None
            self.gt = None
            self.valid_indices = []
            self.label_map = {}

        # Use provided indices if specified, otherwise use all valid indices
        if indices is not None:
            self.indices = indices
        else:
            self.indices = self.valid_indices

    def _load_data(
        self, file_name: str, gt_name: str, image_key: str, gt_key: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        # ... (Keep existing implementation) ...
        image_path = os.path.join(self.data_root, file_name)
        gt_path = os.path.join(self.data_root, gt_name)

        def load_mat_flexible(path: str, key: str, is_gt: bool = False) -> np.ndarray:
            try:
                data = loadmat(path)
                arr = data[key]
            except (NotImplementedError, ValueError):
                print(
                    f"  -> File {os.path.basename(path)} is v7.3 MAT. Using h5py loader..."
                )
                with h5py.File(path, "r") as f:
                    if key not in f:
                        raise KeyError(
                            f"Key '{key}' not found in {path}. Available keys: {list(f.keys())}"
                        )
                    arr = np.array(f[key])
                    if arr.ndim == 3:
                        arr = arr.transpose(2, 1, 0)
                    elif arr.ndim == 2:
                        arr = arr.transpose(1, 0)
            if is_gt:
                return arr.astype(np.int64)
            else:
                return arr.astype(np.float32)

        image = load_mat_flexible(image_path, image_key, is_gt=False)
        gt = load_mat_flexible(gt_path, gt_key, is_gt=True)
        if gt.ndim == 3:
            gt = gt.squeeze()
        return image, gt

    def _apply_pca(self, image: np.ndarray, n_components: int) -> np.ndarray:
        # ... (Keep existing implementation) ...
        H, W, C = image.shape
        if C == n_components:
            return image
        image_2d = image.reshape(-1, C)
        scaler = StandardScaler()
        image_2d = scaler.fit_transform(image_2d)
        pca = PCA(n_components=n_components)
        image_pca = pca.fit_transform(image_2d)
        image_pca = image_pca.reshape(H, W, n_components)
        print(
            f"PCA: Reduced {C} bands to {n_components} bands (explained variance: {pca.explained_variance_ratio_.sum():.4f})"
        )
        return image_pca.astype(np.float32)

    def _pad_image(self, image: np.ndarray) -> np.ndarray:
        # ... (Keep existing implementation) ...
        pad_width = ((self.padding, self.padding), (self.padding, self.padding), (0, 0))
        padded = np.pad(image, pad_width, mode="edge")
        return padded

    def _get_valid_indices(self) -> List[int]:
        # ... (Keep existing implementation) ...
        if self.gt is None:
            return []
        valid_mask = np.ones_like(self.gt, dtype=bool)
        for label in self.ignored_labels:
            valid_mask &= self.gt != label
        valid_indices = np.where(valid_mask.ravel())[0].tolist()
        return valid_indices

    def _create_label_map(self) -> Dict[int, int]:
        # ... (Keep existing implementation) ...
        if self.gt is None:
            return {}
        unique_labels = np.unique(self.gt)
        valid_labels = [l for l in unique_labels if l not in self.ignored_labels]
        valid_labels = sorted(valid_labels)
        label_map = {label: idx for idx, label in enumerate(valid_labels)}
        return label_map

    def _extract_patch(self, x: int, y: int) -> torch.Tensor:
        # ... (Keep existing implementation) ...
        x_padded = x + self.padding
        y_padded = y + self.padding
        patch = self.image[
            x_padded - self.padding : x_padded + self.padding + 1,
            y_padded - self.padding : y_padded + self.padding + 1,
            :,
        ]
        patch = torch.from_numpy(patch).permute(2, 0, 1)
        return patch

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # ... (Keep existing implementation) ...
        linear_idx = self.indices[idx]
        H, W = self.gt.shape
        x = linear_idx // W
        y = linear_idx % W
        original_label = self.gt[x, y]
        label = self.label_map[original_label]
        patch = self._extract_patch(x, y)
        return patch, label

    def __len__(self) -> int:
        return len(self.indices)

    def get_num_classes(self) -> int:
        return len(self.label_map)

    def get_class_counts(self) -> Dict[int, int]:
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
    strategy: str = "spatial_sort",  # New parameter to control split strategy
) -> Tuple[HSIDataset, HSIDataset, HSIDataset]:
    """
    Split dataset into train/val/test sets using Spatial Sorting to prevent leakage.

    Instead of random shuffling, we sort pixels spatially and split the sorted list.
    This ensures train and test samples come from disjoint regions.

    Args:
        dataset: The full dataset
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        seed: Random seed (used for class shuffling if needed, but not pixel shuffling)
        strategy: 'random' (legacy) or 'spatial_sort' (recommended for HSI)

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    np.random.seed(seed)
    print(f"Data Split Strategy: {strategy.upper()}")

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

    for class_idx, indices in class_indices.items():
        indices = np.array(indices)

        if strategy == "spatial_sort":
            # SPATIAL SPLIT: Sort by spatial coordinate (Major axis: X, Minor axis: Y)
            # This keeps spatially adjacent pixels together in the list
            # Note: indices are linear = x * W + y, so standard sort works perfectly for row-major order
            indices = np.sort(indices)

            # Note: We do NOT shuffle here. We rely on the spatial sorting.

        elif strategy == "random":
            # LEGACY: Random shuffle (Prone to leakage!)
            np.random.shuffle(indices)

        n_samples = len(indices)
        n_train = max(1, int(n_samples * train_ratio))
        n_val = max(1, int(n_samples * val_ratio))

        # For spatial sort, this effectively takes the "top" part of the class cluster for train,
        # the "middle" for val, and the "bottom" for test.
        train_indices.extend(indices[:n_train].tolist())
        val_indices.extend(indices[n_train : n_train + n_val].tolist())
        test_indices.extend(indices[n_train + n_val :].tolist())

    # Create dataset objects (Rest of the function remains the same)
    train_dataset = HSIDataset(
        data_root=dataset.data_root,
        file_name=None,
        gt_name=None,
        image_key=None,
        gt_key=None,
        patch_size=dataset.patch_size,
        target_bands=dataset.target_bands,
        ignored_labels=dataset.ignored_labels,
        indices=train_indices,
    )
    # Copy loaded data
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
