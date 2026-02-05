"""Dataset and episodic sampler for few-shot learning.

This module implements PyTorch datasets and samplers for episodic
training in the few-shot learning paradigm.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Dict, Optional
import random


class HSIFewShotDataset(Dataset):
    """Dataset for HSI few-shot learning.
    
    This dataset stores preprocessed patches and their labels,
    organized by class for episodic sampling.
    """
    
    def __init__(
        self,
        patches: np.ndarray,
        labels: np.ndarray,
        class_ids: Optional[List[int]] = None
    ):
        """Initialize the few-shot dataset.
        
        Args:
            patches: Array of shape (N, patch_size, patch_size, bands).
            labels: Array of shape (N,) with class labels.
            class_ids: List of class IDs to include. If None, uses all classes.
        """
        self.patches = patches
        self.labels = labels
        
        # Organize data by class
        self.class_to_indices: Dict[int, List[int]] = {}
        unique_labels = np.unique(labels)
        
        # Filter by class_ids if provided
        if class_ids is not None:
            unique_labels = [c for c in unique_labels if c in class_ids]
        
        for class_id in unique_labels:
            indices = np.where(labels == class_id)[0].tolist()
            if len(indices) > 0:
                self.class_to_indices[class_id] = indices
        
        self.num_classes = len(self.class_to_indices)
        self.class_list = sorted(self.class_to_indices.keys())
        
        print(f"Dataset created with {len(patches)} samples across {self.num_classes} classes")
        for class_id in self.class_list:
            print(f"  Class {class_id}: {len(self.class_to_indices[class_id])} samples")
    
    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.patches)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample.
        
        Args:
            idx: Sample index.
            
        Returns:
            Tuple of (patch, label) where patch is a torch.Tensor.
        """
        patch = self.patches[idx]
        label = self.labels[idx]
        
        # Convert to tensor and add channel dimension
        # Shape: (1, D, H, W) where D is spectral depth
        patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0)
        
        return patch_tensor, int(label)
    
    def get_class_samples(
        self,
        class_id: int,
        num_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get random samples from a specific class.
        
        Args:
            class_id: Class to sample from.
            num_samples: Number of samples to retrieve.
            
        Returns:
            Tuple of (patches, labels) as torch.Tensors.
            
        Raises:
            ValueError: If class_id not in dataset or insufficient samples.
        """
        if class_id not in self.class_to_indices:
            raise ValueError(f"Class {class_id} not in dataset")
        
        available_indices = self.class_to_indices[class_id]
        
        if len(available_indices) < num_samples:
            raise ValueError(
                f"Class {class_id} has only {len(available_indices)} samples, "
                f"but {num_samples} requested"
            )
        
        # Sample without replacement
        selected_indices = random.sample(available_indices, num_samples)
        
        patches = []
        labels = []
        
        for idx in selected_indices:
            patch, label = self[idx]
            patches.append(patch)
            labels.append(label)
        
        return torch.stack(patches), torch.tensor(labels)


class EpisodeSampler:
    """Episodic sampler for few-shot learning.
    
    Generates episodes with support and query sets for N-way K-shot learning.
    Ensures no data leakage between support and query sets.
    """
    
    def __init__(
        self,
        dataset: HSIFewShotDataset,
        n_way: int = 5,
        k_shot: int = 5,
        q_query: int = 15,
        num_episodes: int = 100,
        random_seed: Optional[int] = None
    ):
        """Initialize the episodic sampler.
        
        Args:
            dataset: HSIFewShotDataset to sample from.
            n_way: Number of classes per episode.
            k_shot: Number of support samples per class.
            q_query: Number of query samples per class.
            num_episodes: Total number of episodes to generate.
            random_seed: Random seed for reproducibility.
        """
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.num_episodes = num_episodes
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Validate configuration
        if self.n_way > dataset.num_classes:
            raise ValueError(
                f"n_way ({n_way}) cannot exceed number of classes "
                f"in dataset ({dataset.num_classes})"
            )
        
        # Check if all classes have enough samples
        min_samples_needed = k_shot + q_query
        for class_id in dataset.class_list:
            num_samples = len(dataset.class_to_indices[class_id])
            if num_samples < min_samples_needed:
                print(
                    f"Warning: Class {class_id} has only {num_samples} samples, "
                    f"but {min_samples_needed} needed for k_shot={k_shot} + "
                    f"q_query={q_query}"
                )
    
    def __len__(self) -> int:
        """Return number of episodes."""
        return self.num_episodes
    
    def __iter__(self):
        """Iterator for generating episodes."""
        for _ in range(self.num_episodes):
            yield self.sample_episode()
    
    def sample_episode(self) -> Dict[str, torch.Tensor]:
        """Sample a single episode.
        
        Returns:
            Dictionary containing:
                - support_data: Tensor of shape (n_way * k_shot, 1, D, H, W)
                - support_labels: Tensor of shape (n_way * k_shot,)
                - query_data: Tensor of shape (n_way * q_query, 1, D, H, W)
                - query_labels: Tensor of shape (n_way * q_query,)
                - class_ids: List of selected class IDs for this episode
        """
        # Randomly select n_way classes
        selected_classes = random.sample(self.dataset.class_list, self.n_way)
        
        support_data_list = []
        support_labels_list = []
        query_data_list = []
        query_labels_list = []
        
        # Create a mapping from original class IDs to episode labels (0 to n_way-1)
        class_to_episode_label = {
            class_id: episode_label 
            for episode_label, class_id in enumerate(selected_classes)
        }
        
        for class_id in selected_classes:
            # Get all indices for this class
            all_indices = self.dataset.class_to_indices[class_id].copy()
            
            # Shuffle and split into support and query
            random.shuffle(all_indices)
            
            support_indices = all_indices[:self.k_shot]
            query_indices = all_indices[self.k_shot:self.k_shot + self.q_query]
            
            # Get support samples
            for idx in support_indices:
                patch, _ = self.dataset[idx]
                support_data_list.append(patch)
                support_labels_list.append(class_to_episode_label[class_id])
            
            # Get query samples
            for idx in query_indices:
                patch, _ = self.dataset[idx]
                query_data_list.append(patch)
                query_labels_list.append(class_to_episode_label[class_id])
        
        # Stack into tensors
        support_data = torch.stack(support_data_list)
        support_labels = torch.tensor(support_labels_list, dtype=torch.long)
        query_data = torch.stack(query_data_list)
        query_labels = torch.tensor(query_labels_list, dtype=torch.long)
        
        return {
            'support_data': support_data,
            'support_labels': support_labels,
            'query_data': query_data,
            'query_labels': query_labels,
            'class_ids': selected_classes
        }
    
    def get_episode_batch(self, batch_size: int = 1) -> List[Dict[str, torch.Tensor]]:
        """Get a batch of episodes.
        
        Args:
            batch_size: Number of episodes to generate.
            
        Returns:
            List of episode dictionaries.
        """
        return [self.sample_episode() for _ in range(batch_size)]
