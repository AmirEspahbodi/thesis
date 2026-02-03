"""
Few-Shot Episodic Sampler Module

Implements sampling strategy for few-shot learning episodes (tasks).
Each episode contains:
- N classes (n_way)
- K support samples per class (k_shot)
- Q query samples per class (query_shot)
"""

from functools import partial
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class FewShotSampler(Sampler):
    """
    Episodic sampler for few-shot learning
    """

    def __init__(
        self,
        dataset: Dataset,
        n_way: int = 5,
        k_shot: int = 5,
        query_shot: int = 15,
        n_episodes: int = 100,
        deterministic: bool = False,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_shot = query_shot
        self.n_episodes = n_episodes
        self.deterministic = deterministic
        self.seed = seed

        # Get class-to-indices mapping
        self.class_indices = self._group_by_class()
        self._validate_class_samples()
        self.available_classes = list(self.class_indices.keys())

    def _group_by_class(self) -> Dict[int, List[int]]:
        class_indices = {}
        for idx in range(len(self.dataset)):
            _, label = self.dataset[idx]
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        return class_indices

    def _validate_class_samples(self):
        min_required = self.k_shot + self.query_shot
        for class_idx, indices in self.class_indices.items():
            if len(indices) < min_required:
                raise ValueError(
                    f"Class {class_idx} has only {len(indices)} samples, "
                    f"but needs at least {min_required} (k_shot={self.k_shot} + query_shot={self.query_shot})"
                )

    def __iter__(self) -> Iterator[List[int]]:
        # If deterministic, create a local random state
        rng = np.random.RandomState(self.seed) if self.deterministic else np.random

        for _ in range(self.n_episodes):
            episode_classes = rng.choice(
                self.available_classes, size=self.n_way, replace=False
            )

            support_indices = []
            query_indices = []

            for class_idx in episode_classes:
                class_samples = self.class_indices[class_idx].copy()

                # Use the appropriate RNG for shuffling
                rng.shuffle(class_samples)

                support = class_samples[: self.k_shot]
                query = class_samples[self.k_shot : self.k_shot + self.query_shot]

                support_indices.extend(support)
                query_indices.extend(query)

            # CRITICAL: Support must come first, then query
            episode_indices = support_indices + query_indices
            yield episode_indices

    def __len__(self) -> int:
        return self.n_episodes


def collate_few_shot_batch(
    batch: List[Tuple[torch.Tensor, int]], n_way: int, k_shot: int, query_shot: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for few-shot learning with CORRECT support/query split.

    FIX #1: Uses actual n_way × k_shot formula instead of incorrect 50/50 split.

    Args:
        batch: List of (patch, label) tuples from an episode
        n_way: Number of classes in the episode
        k_shot: Number of support samples per class
        query_shot: Number of query samples per class

    Returns:
        support_patches: (n_way * k_shot, C, H, W) tensor
        support_labels: (n_way * k_shot,) tensor
        query_patches: (n_way * query_shot, C, H, W) tensor
        query_labels: (n_way * query_shot,) tensor

    Raises:
        AssertionError: If batch size doesn't match expected episode size

    Note:
        The batch MUST be organized with support samples first (n_way × k_shot),
        then query samples (n_way × query_shot), as returned by FewShotSampler.

    Edge Cases Handled:
        - Validates batch size matches n_way × (k_shot + query_shot)
        - Ensures no overlap between support and query indices
        - Handles various n_way, k_shot, query_shot configurations
    """
    # Step 1: Extract patches and labels
    patches, labels = zip(*batch)
    patches = torch.stack(patches)  # (N, C, H, W)
    labels = torch.tensor(labels)  # (N,)

    # Step 2: Calculate correct split point using episode configuration
    n_support = n_way * k_shot
    n_query = n_way * query_shot
    expected_total = n_support + n_query
    actual_total = len(patches)

    # Step 3: Validate batch size
    assert actual_total == expected_total, (
        f"Batch size mismatch! Expected {expected_total} samples "
        f"({n_way} ways × ({k_shot} support + {query_shot} query)), "
        f"but got {actual_total} samples. "
        f"This indicates a bug in the episode sampler."
    )

    # Step 4: Split at the correct position (NOT len//2)
    support_patches = patches[:n_support]
    support_labels = labels[:n_support]
    query_patches = patches[n_support:]
    query_labels = labels[n_support:]

    # Step 5: Post-condition verification
    assert len(support_patches) == n_support, (
        f"Support size mismatch: {len(support_patches)} != {n_support}"
    )
    assert len(query_patches) == n_query, (
        f"Query size mismatch: {len(query_patches)} != {n_query}"
    )

    return support_patches, support_labels, query_patches, query_labels


def create_collate_fn(n_way: int, k_shot: int, query_shot: int):
    """
    Factory function to create a collate function with injected episode parameters.

    This allows DataLoader to use the correct split without hardcoding.

    Example:
        >>> collate_fn = create_collate_fn(n_way=5, k_shot=5, query_shot=15)
        >>> loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)

    Args:
        n_way: Number of classes per episode
        k_shot: Support samples per class
        query_shot: Query samples per class

    Returns:
        Collate function with parameters bound via partial()
    """
    return partial(
        collate_few_shot_batch, n_way=n_way, k_shot=k_shot, query_shot=query_shot
    )


class EpisodeBatchSampler:
    """
    Batch sampler that groups multiple episodes together

    Useful for processing multiple few-shot tasks in parallel

    Args:
        sampler: FewShotSampler instance
        batch_size: Number of episodes per batch
    """

    def __init__(self, sampler: FewShotSampler, batch_size: int = 1):
        self.sampler = sampler
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[List[List[int]]]:
        """
        Generate batches of episodes

        Yields:
            List of episode index lists
        """
        batch = []
        for episode_indices in self.sampler:
            batch.append(episode_indices)

            if len(batch) == self.batch_size:
                yield batch
                batch = []

        # Yield remaining episodes if any
        if len(batch) > 0:
            yield batch

    def __len__(self) -> int:
        """Return number of batches"""
        return len(self.sampler) // self.batch_size
