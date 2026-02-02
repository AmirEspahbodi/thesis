"""
Few-Shot Episodic Sampler Module

Implements sampling strategy for few-shot learning episodes (tasks).
Each episode contains:
- N classes (n_way)
- K support samples per class (k_shot)
- Q query samples per class (query_shot)
"""

from typing import Dict, Iterator, List

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class FewShotSampler(Sampler):
    """
    Episodic sampler for few-shot learning
    ...
    """

    def __init__(
        self,
        dataset: Dataset,
        n_way: int = 5,
        k_shot: int = 5,
        query_shot: int = 15,
        n_episodes: int = 100,
        deterministic: bool = False,  # NEW
        seed: int = 42,  # NEW
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
                    f"but needs at least {min_required} "
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

            episode_indices = support_indices + query_indices
            yield episode_indices

    def __len__(self) -> int:
        return self.n_episodes


def collate_few_shot_batch(batch: List[tuple]) -> tuple:
    """
    Custom collate function for few-shot learning

    Organizes batch into support and query sets

    Args:
        batch: List of (patch, label) tuples from an episode

    Returns:
        support_patches: (n_way * k_shot, C, H, W) tensor
        support_labels: (n_way * k_shot,) tensor
        query_patches: (n_way * query_shot, C, H, W) tensor
        query_labels: (n_way * query_shot,) tensor

    Note:
        The batch is assumed to be organized with support samples first,
        then query samples, as returned by FewShotSampler
    """
    patches, labels = zip(*batch)

    patches = torch.stack(patches)  # (N, C, H, W)
    labels = torch.tensor(labels)  # (N,)

    # The first half are support samples, second half are query samples
    n_samples = len(patches)
    n_support = n_samples // 2

    support_patches = patches[:n_support]
    support_labels = labels[:n_support]
    query_patches = patches[n_support:]
    query_labels = labels[n_support:]

    return support_patches, support_labels, query_patches, query_labels


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
