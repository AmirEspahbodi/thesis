"""Prototypical Network for few-shot classification.

This module implements the Prototypical Network meta-learning algorithm
for few-shot classification of hyperspectral images.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from .backbone import HSI3DCNN


class PrototypicalNetwork(nn.Module):
    """Prototypical Network for few-shot learning.
    
    This network computes class prototypes from support set embeddings
    and classifies query samples based on their distance to prototypes.
    """
    
    def __init__(self, backbone: nn.Module):
        """Initialize the Prototypical Network.
        
        Args:
            backbone: Feature extractor network (e.g., HSI3DCNN).
        """
        super(PrototypicalNetwork, self).__init__()
        self.backbone = backbone
    
    def compute_prototypes(
        self,
        support_embeddings: torch.Tensor,
        support_labels: torch.Tensor,
        n_way: int
    ) -> torch.Tensor:
        """Compute class prototypes from support set.
        
        Class prototypes are computed as the mean of all support
        embeddings for each class.
        
        Args:
            support_embeddings: Embeddings of shape (n_support, embedding_dim).
            support_labels: Labels of shape (n_support,).
            n_way: Number of classes.
            
        Returns:
            Prototypes of shape (n_way, embedding_dim).
        """
        prototypes = torch.zeros(
            n_way,
            support_embeddings.size(1),
            device=support_embeddings.device
        )
        
        for class_idx in range(n_way):
            # Get all embeddings for this class
            class_mask = support_labels == class_idx
            class_embeddings = support_embeddings[class_mask]
            
            # Compute mean (prototype)
            prototypes[class_idx] = class_embeddings.mean(dim=0)
        
        return prototypes
    
    def compute_distances(
        self,
        query_embeddings: torch.Tensor,
        prototypes: torch.Tensor,
        distance_metric: str = "euclidean"
    ) -> torch.Tensor:
        """Compute distances from query embeddings to prototypes.
        
        Args:
            query_embeddings: Embeddings of shape (n_query, embedding_dim).
            prototypes: Prototypes of shape (n_way, embedding_dim).
            distance_metric: Distance metric to use ('euclidean' or 'cosine').
            
        Returns:
            Distance matrix of shape (n_query, n_way).
        """
        if distance_metric == "euclidean":
            # Compute squared Euclidean distances
            # Using broadcasting: (n_query, 1, dim) - (1, n_way, dim)
            distances = torch.sum(
                (query_embeddings.unsqueeze(1) - prototypes.unsqueeze(0)) ** 2,
                dim=2
            )
        elif distance_metric == "cosine":
            # Compute cosine similarity
            query_norm = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
            proto_norm = torch.nn.functional.normalize(prototypes, p=2, dim=1)
            distances = -torch.mm(query_norm, proto_norm.t())  # Negative for similarity
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")
        
        return distances
    
    def forward(
        self,
        support_data: torch.Tensor,
        support_labels: torch.Tensor,
        query_data: torch.Tensor,
        n_way: int,
        distance_metric: str = "euclidean"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for an episode.
        
        Args:
            support_data: Support set data of shape (n_support, C, D, H, W).
            support_labels: Support labels of shape (n_support,).
            query_data: Query set data of shape (n_query, C, D, H, W).
            n_way: Number of classes in this episode.
            distance_metric: Distance metric to use.
            
        Returns:
            Tuple of (logits, prototypes):
                - logits: Classification scores of shape (n_query, n_way).
                - prototypes: Class prototypes of shape (n_way, embedding_dim).
        """
        # Extract features for support set
        support_embeddings = self.backbone(support_data)
        
        # Extract features for query set
        query_embeddings = self.backbone(query_data)
        
        # Compute class prototypes
        prototypes = self.compute_prototypes(support_embeddings, support_labels, n_way)
        
        # Compute distances (negative distances as logits)
        distances = self.compute_distances(query_embeddings, prototypes, distance_metric)
        
        # Convert distances to logits (negative distances for softmax)
        logits = -distances
        
        return logits, prototypes
    
    def predict(
        self,
        support_data: torch.Tensor,
        support_labels: torch.Tensor,
        query_data: torch.Tensor,
        n_way: int,
        distance_metric: str = "euclidean"
    ) -> torch.Tensor:
        """Predict class labels for query samples.
        
        Args:
            support_data: Support set data.
            support_labels: Support labels.
            query_data: Query set data.
            n_way: Number of classes.
            distance_metric: Distance metric to use.
            
        Returns:
            Predicted class labels of shape (n_query,).
        """
        with torch.no_grad():
            logits, _ = self.forward(
                support_data, support_labels, query_data, n_way, distance_metric
            )
            predictions = torch.argmax(logits, dim=1)
        
        return predictions
    
    def extract_features(self, data: torch.Tensor) -> torch.Tensor:
        """Extract features using the backbone.
        
        Args:
            data: Input data of shape (B, C, D, H, W).
            
        Returns:
            Features of shape (B, embedding_dim).
        """
        return self.backbone(data)


def build_prototypical_network(
    input_channels: int = 1,
    spectral_depth: int = 30,
    spatial_size: int = 9,
    conv_channels: Tuple[int, ...] = (32, 64, 128),
    kernel_sizes: Tuple[Tuple[int, int, int], ...] = ((3, 3, 3), (3, 3, 3), (3, 3, 3)),
    pool_sizes: Tuple[Tuple[int, int, int], ...] = ((2, 2, 2), (2, 2, 2), (2, 2, 2)),
    embedding_dim: int = 256,
    dropout_rate: float = 0.3
) -> PrototypicalNetwork:
    """Build a complete Prototypical Network with HSI3DCNN backbone.
    
    Args:
        input_channels: Number of input channels.
        spectral_depth: Spectral dimension after PCA.
        spatial_size: Spatial size of patches.
        conv_channels: Channels for conv layers.
        kernel_sizes: Kernel sizes for conv layers.
        pool_sizes: Pool sizes for conv layers.
        embedding_dim: Embedding dimension.
        dropout_rate: Dropout rate.
        
    Returns:
        PrototypicalNetwork instance.
    """
    backbone = HSI3DCNN(
        input_channels=input_channels,
        spectral_depth=spectral_depth,
        spatial_size=spatial_size,
        conv_channels=conv_channels,
        kernel_sizes=kernel_sizes,
        pool_sizes=pool_sizes,
        embedding_dim=embedding_dim,
        dropout_rate=dropout_rate
    )
    
    model = PrototypicalNetwork(backbone)
    
    return model
