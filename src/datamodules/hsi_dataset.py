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
    Hyperspectral Image Dataset for Few-Shot Learning.
    Supports loading .mat files, PCA reduction, and spatial patch extraction.
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

        if file_name is not None:
            self.image, self.gt = self._load_data(file_name, gt_name, image_key, gt_key)
            self.image = self._apply_pca(self.image, target_bands)
            self.image = self._pad_image(self.image)
            self.valid_indices = self._get_valid_indices()
            self.label_map = self._create_label_map()
        else:
            self.image = None
            self.gt = None
            self.valid_indices = []
            self.label_map = {}

        self.indices = indices if indices is not None else self.valid_indices

    def _load_data(
        self, file_name: str, gt_name: str, image_key: str, gt_key: str
    ) -> Tuple[np.ndarray, np.ndarray]:
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

    def _apply_pca(self, image: np.ndarray, n_components: int) -> np.ndarray:
        H, W, C = image.shape
        if C == n_components:
            return image
        image_2d = image.reshape(-1, C)
        scaler = StandardScaler()
        image_2d = scaler.fit_transform(image_2d)
        pca = PCA(n_components=n_components)
        image_pca = pca.fit_transform(image_2d)
        return image_pca.reshape(H, W, n_components).astype(np.float32)

    def _pad_image(self, image: np.ndarray) -> np.ndarray:
        pad_width = ((self.padding, self.padding), (self.padding, self.padding), (0, 0))
        return np.pad(image, pad_width, mode="edge")

    def _get_valid_indices(self) -> List[int]:
        if self.gt is None:
            return []
        valid_mask = np.ones_like(self.gt, dtype=bool)
        for label in self.ignored_labels:
            valid_mask &= self.gt != label
        return np.where(valid_mask.ravel())[0].tolist()

    def _create_label_map(self) -> Dict[int, int]:
        if self.gt is None:
            return {}
        unique_labels = np.unique(self.gt)
        valid_labels = sorted(
            [l for l in unique_labels if l not in self.ignored_labels]
        )
        return {label: idx for idx, label in enumerate(valid_labels)}

    def _extract_patch(self, x: int, y: int) -> torch.Tensor:
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


def create_data_splits(
    dataset: HSIDataset,
    train_ratio: float = 0.1,
    val_ratio: float = 0.1,
    seed: int = 42,
    strategy: str = "spatial_sort",
) -> Tuple[HSIDataset, HSIDataset, HSIDataset]:
    """
    Split dataset into train/val/test sets with spatial disjoint buffers to prevent leakage.

    A safety buffer based on patch_size is enforced between sets. A candidate pixel is
    only included if its Chebyshev distance from the previous set is >= patch_size.
    """
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

    print(f"Data Split Strategy: {strategy.upper()} (Buffer Size: {patch_size})")

    for class_idx, indices in class_indices.items():
        indices = np.sort(np.array(indices))

        if strategy == "random":
            np.random.shuffle(indices)
            n_samples = len(indices)
            n_train = max(1, int(n_samples * train_ratio))
            n_val = max(1, int(n_samples * val_ratio))

            train_indices.extend(indices[:n_train].tolist())
            val_indices.extend(indices[n_train : n_train + n_val].tolist())
            test_indices.extend(indices[n_train + n_val :].tolist())
            continue

        # Strategy: spatial_sort with Safety Buffers
        n_samples = len(indices)
        n_train_target = max(1, int(n_samples * train_ratio))
        n_val_target = max(1, int(n_samples * val_ratio))

        # 1. Select Training Set
        c_train = indices[:n_train_target].tolist()
        train_indices.extend(c_train)

        remaining = indices[n_train_target:]
        dropped_val_gap = 0

        # 2. Find start of Validation Set (Gap Search)
        last_train_idx = c_train[-1]
        tx, ty = last_train_idx // W, last_train_idx % W

        val_start_idx = 0
        while val_start_idx < len(remaining):
            vx, vy = remaining[val_start_idx] // W, remaining[val_start_idx] % W
            if max(abs(tx - vx), abs(ty - vy)) >= patch_size:
                break
            val_start_idx += 1
            dropped_val_gap += 1

        remaining = remaining[val_start_idx:]

        # 3. Select Validation Set
        c_val = remaining[:n_val_target].tolist()
        if not c_val:
            print(f"Warning: Class {class_idx} Validation set is empty after buffer.")
        val_indices.extend(c_val)

        remaining = remaining[n_val_target:]
        dropped_test_gap = 0

        # 4. Find start of Test Set (Gap Search)
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

        # 5. Select Test Set
        c_test = remaining.tolist()
        if not c_test:
            print(f"Warning: Class {class_idx} Test set is empty after buffer.")
        test_indices.extend(c_test)

        print(
            f"Class {class_idx:2d}: Dropped {dropped_val_gap:3d} (Val Gap), {dropped_test_gap:3d} (Test Gap) samples."
        )

    # Utility to clone dataset with specific indices
    def _wrap(idx_list):
        ds = HSIDataset(
            dataset.data_root,
            None,
            None,
            None,
            None,
            patch_size,
            dataset.target_bands,
            dataset.ignored_labels,
            idx_list,
        )
        ds.image, ds.gt, ds.label_map, ds.valid_indices = (
            dataset.image,
            dataset.gt,
            dataset.label_map,
            dataset.valid_indices,
        )
        return ds

    train_ds, val_ds, test_ds = (
        _wrap(train_indices),
        _wrap(val_indices),
        _wrap(test_indices),
    )
    print(
        f"Split Summary - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}"
    )

    return train_ds, val_ds, test_ds
