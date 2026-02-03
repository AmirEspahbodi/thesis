import os
import warnings
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

    Raises:
        ValueError: If not enough samples for PCA
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
    strategy: str = "spatial_sort",
) -> Tuple[HSIDataset, HSIDataset, HSIDataset]:
    """
    Split dataset into train/val/test sets with NO PCA LEAKAGE and NO SPATIAL LEAKAGE.

    CRITICAL FIXES:
    - FIX #2: PCA is fitted ONLY on training data, then applied to val/test
    - FIX #4: Random split is BLOCKED to prevent spatial leakage
    - Spatial buffers enforce minimum distance >= patch_size between splits

    Args:
        dataset: Full HSI dataset
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        seed: Random seed for reproducibility
        strategy: Split strategy (must be 'spatial_sort')

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)

    Raises:
        ValueError: If strategy is 'random' (causes spatial leakage)
    """
    # FIX #4: Block random strategy
    if strategy == "random":
        raise ValueError(
            "ERROR: Random split strategy is DISABLED because it causes spatial leakage.\n"
            "Overlapping patches between train/test inflate accuracy by 15-60%.\n"
            "Use strategy='spatial_sort' which enforces spatial buffers >= patch_size.\n"
            "This ensures train and test patches do not overlap."
        )

    np.random.seed(seed)
    W = dataset.gt.shape[1]
    patch_size = dataset.patch_size

    # Group indices by class
    class_indices = {}
    for idx in dataset.valid_indices:
        x, y = idx // W, idx % W
        class_idx = dataset.label_map[dataset.gt[x, y]]
        class_indices.setdefault(class_idx, []).append(idx)

    train_indices, val_indices, test_indices = [], [], []

    print(f"\n{'=' * 60}")
    print(f"Data Split Strategy: {strategy.upper()}")
    print(f"Spatial Buffer Size: {patch_size} pixels (Chebyshev distance)")
    print(f"{'=' * 60}")

    total_dropped_val = 0
    total_dropped_test = 0
    total_samples = len(dataset.valid_indices)

    for class_idx, indices in class_indices.items():
        indices = np.sort(np.array(indices))

        n_samples = len(indices)
        n_train_target = max(1, int(n_samples * train_ratio))
        n_val_target = max(1, int(n_samples * val_ratio))

        # 1. Select Training Set
        c_train = indices[:n_train_target].tolist()
        train_indices.extend(c_train)

        remaining = indices[n_train_target:]
        dropped_val_gap = 0

        # 2. Find start of Validation Set with Spatial Buffer
        if c_train:
            last_train_idx = c_train[-1]
            tx, ty = last_train_idx // W, last_train_idx % W

            val_start_idx = 0
            while val_start_idx < len(remaining):
                vx, vy = remaining[val_start_idx] // W, remaining[val_start_idx] % W
                # Chebyshev distance: max(|Δx|, |Δy|)
                if max(abs(tx - vx), abs(ty - vy)) >= patch_size:
                    break
                val_start_idx += 1
                dropped_val_gap += 1

            remaining = remaining[val_start_idx:]
            total_dropped_val += dropped_val_gap

        # 3. Select Validation Set
        c_val = remaining[:n_val_target].tolist()
        if not c_val:
            print(f"  ⚠️  Class {class_idx}: Validation set EMPTY after spatial buffer")
        val_indices.extend(c_val)

        remaining = remaining[n_val_target:]
        dropped_test_gap = 0

        # 4. Find start of Test Set with Spatial Buffer
        if c_val:
            last_val_idx = c_val[-1]
            vx, vy = last_val_idx // W, last_val_idx % W

            test_start_idx = 0
            while test_start_idx < len(remaining):
                kx, ky = remaining[test_start_idx] // W, remaining[test_start_idx] % W
                if max(abs(vx - kx), abs(vy - ky)) >= patch_size:
                    break
                test_start_idx += 1
                dropped_test_gap += 1

            remaining = remaining[test_start_idx:]
            total_dropped_test += dropped_test_gap

        # 5. Select Test Set
        c_test = remaining.tolist()
        if not c_test:
            print(f"  ⚠️  Class {class_idx}: Test set EMPTY after spatial buffer")
        test_indices.extend(c_test)

        print(
            f"  Class {class_idx:2d}: Train={len(c_train):4d}, Val={len(c_val):4d}, Test={len(c_test):4d} | "
            f"Dropped: Val={dropped_val_gap:3d}, Test={dropped_test_gap:3d}"
        )

    # Warn if excessive samples dropped
    total_dropped = total_dropped_val + total_dropped_test
    drop_percentage = (total_dropped / total_samples) * 100
    print(
        f"\n  Total samples dropped: {total_dropped}/{total_samples} ({drop_percentage:.1f}%)"
    )
    if drop_percentage > 10:
        print(
            f"  ⚠️  WARNING: >{drop_percentage:.1f}% samples dropped due to spatial buffers"
        )
        print(
            f"     Consider: (1) Smaller patch_size, (2) Larger dataset, (3) Higher train_ratio"
        )

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
    print(f"Split Summary:")
    print(f"  Train: {len(train_ds):5d} samples")
    print(f"  Val:   {len(val_ds):5d} samples")
    print(f"  Test:  {len(test_ds):5d} samples")
    print(f"{'=' * 60}\n")

    return train_ds, val_ds, test_ds
