"""
Prototypical Network for Few-Shot Learning

Implements the Prototypical Networks algorithm:
1. Extract features from support and query sets
2. Compute class prototypes (mean of support features per class)
3. Classify queries based on distance to prototypes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PrototypicalNetwork(nn.Module):
    """
    Prototypical Network for Few-Shot Classification

    The network:
    1. Uses a backbone to extract features
    2. Computes prototypes (class centers) from support set
    3. Classifies query samples based on Euclidean distance to prototypes

    Args:
        backbone: Feature extractor (e.g., Simple3DCNN)
        distance_metric: Distance metric ('euclidean' or 'cosine')
    """

    def __init__(
        self,
        backbone: nn.Module,
        distance_metric: str = "euclidean",
    ):
        super().__init__()

        self.backbone = backbone
        self.distance_metric = distance_metric

        if distance_metric not in ["euclidean", "cosine"]:
            raise ValueError(f"Unknown distance metric: {distance_metric}")

    def compute_prototypes(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        n_way: int,
    ) -> torch.Tensor:
        """
        Compute class prototypes from support set

        Args:
            support_features: (n_way * k_shot, d_model) tensor
            support_labels: (n_way * k_shot,) tensor with values in [0, n_way)
            n_way: Number of classes

        Returns:
            prototypes: (n_way, d_model) tensor
        """
        prototypes = []

        for class_idx in range(n_way):
            # Get features for this class
            class_mask = support_labels == class_idx
            class_features = support_features[class_mask]

            # Compute mean (prototype) for this class
            prototype = class_features.mean(dim=0)
            prototypes.append(prototype)

        prototypes = torch.stack(prototypes)  # (n_way, d_model)

        return prototypes

    def compute_distances(
        self,
        query_features: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute distances between query features and prototypes

        Args:
            query_features: (n_query, d_model) tensor
            prototypes: (n_way, d_model) tensor

        Returns:
            distances: (n_query, n_way) tensor
        """
        if self.distance_metric == "euclidean":
            # Euclidean distance: ||q - p||^2
            # Expand dimensions for broadcasting
            # query_features: (n_query, 1, d_model)
            # prototypes: (1, n_way, d_model)
            query_expanded = query_features.unsqueeze(1)
            proto_expanded = prototypes.unsqueeze(0)

            # Compute squared Euclidean distance
            distances = torch.sum(
                (query_expanded - proto_expanded) ** 2, dim=2
            )  # (n_query, n_way)

        elif self.distance_metric == "cosine":
            # Cosine similarity (higher is better, so negate)
            # Normalize features
            query_norm = F.normalize(query_features, p=2, dim=1)
            proto_norm = F.normalize(prototypes, p=2, dim=1)

            # Compute cosine similarity
            similarity = torch.mm(query_norm, proto_norm.t())

            # Convert to distance (negate so smaller is better)
            distances = -similarity

        return distances

    def forward(
        self,
        support_patches: torch.Tensor,
        support_labels: torch.Tensor,
        query_patches: torch.Tensor,
        query_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for Prototypical Network

        Args:
            support_patches: (n_way * k_shot, C, H, W) tensor
            support_labels: (n_way * k_shot,) tensor with values in [0, n_way)
            query_patches: (n_way * query_shot, C, H, W) tensor
            query_labels: Optional (n_way * query_shot,) tensor for loss computation

        Returns:
            logits: (n_query, n_way) tensor (negative distances)
            loss: Scalar tensor (if query_labels provided, else None)
            predictions: (n_query,) tensor with predicted class indices
        """
        # Extract features
        support_features = self.backbone(support_patches)  # (n_support, d_model)
        query_features = self.backbone(query_patches)  # (n_query, d_model)

        # Determine n_way from support labels
        n_way = len(torch.unique(support_labels))

        # Compute prototypes
        prototypes = self.compute_prototypes(
            support_features, support_labels, n_way
        )  # (n_way, d_model)

        # Compute distances
        distances = self.compute_distances(
            query_features, prototypes
        )  # (n_query, n_way)

        # Convert distances to logits (negative distances)
        logits = -distances

        # Predictions
        predictions = torch.argmax(logits, dim=1)

        # Compute loss if labels provided
        loss = None
        if query_labels is not None:
            loss = F.cross_entropy(logits, query_labels)

        return logits, loss, predictions

    def predict(
        self,
        support_patches: torch.Tensor,
        support_labels: torch.Tensor,
        query_patches: torch.Tensor,
    ) -> torch.Tensor:
        """
        Make predictions without computing loss

        Args:
            support_patches: (n_way * k_shot, C, H, W) tensor
            support_labels: (n_way * k_shot,) tensor
            query_patches: (n_way * query_shot, C, H, W) tensor

        Returns:
            predictions: (n_query,) tensor with predicted class indices
        """
        with torch.no_grad():
            _, _, predictions = self.forward(
                support_patches, support_labels, query_patches, query_labels=None
            )

        return predictions


class PrototypicalNetworkWithAttention(nn.Module):
    """
    Enhanced Prototypical Network with Attention Mechanism

    This is a placeholder for future attention-based enhancements.
    Can be implemented later to add:
    - Self-attention over support samples
    - Cross-attention between query and support
    - Channel attention for feature refinement
    """

    def __init__(
        self,
        backbone: nn.Module,
        d_model: int,
        n_heads: int = 4,
        distance_metric: str = "euclidean",
    ):
        super().__init__()

        self.backbone = backbone
        self.distance_metric = distance_metric

        # Multi-head attention for support set
        self.support_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )

        # Layer normalization
        self.ln = nn.LayerNorm(d_model)

    def forward(
        self,
        support_patches: torch.Tensor,
        support_labels: torch.Tensor,
        query_patches: torch.Tensor,
        query_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention mechanism

        TODO: Implement attention-based prototype refinement
        """
        # Extract features
        support_features = self.backbone(support_patches)
        query_features = self.backbone(query_patches)

        # Apply self-attention to support features
        # This can help the model focus on discriminative features
        support_features_attn, _ = self.support_attention(
            support_features.unsqueeze(0),
            support_features.unsqueeze(0),
            support_features.unsqueeze(0),
        )
        support_features_attn = support_features_attn.squeeze(0)

        # Residual connection and normalization
        support_features = self.ln(support_features + support_features_attn)

        # Rest is same as standard ProtoNet
        n_way = len(torch.unique(support_labels))

        prototypes = []
        for class_idx in range(n_way):
            class_mask = support_labels == class_idx
            class_features = support_features[class_mask]
            prototype = class_features.mean(dim=0)
            prototypes.append(prototype)

        prototypes = torch.stack(prototypes)

        # Compute distances
        query_expanded = query_features.unsqueeze(1)
        proto_expanded = prototypes.unsqueeze(0)
        distances = torch.sum((query_expanded - proto_expanded) ** 2, dim=2)

        logits = -distances
        predictions = torch.argmax(logits, dim=1)

        loss = None
        if query_labels is not None:
            loss = F.cross_entropy(logits, query_labels)

        return logits, loss, predictions


if __name__ == "__main__":
    # Test the Prototypical Network
    from backbone import Simple3DCNN

    # Create backbone
    backbone = Simple3DCNN(
        input_channels=1, spectral_size=30, spatial_size=9, d_model=128
    )

    # Create ProtoNet
    model = PrototypicalNetwork(backbone)

    # Test data (5-way 5-shot)
    n_way = 5
    k_shot = 5
    query_shot = 15

    support_patches = torch.randn(n_way * k_shot, 30, 9, 9)
    support_labels = torch.repeat_interleave(torch.arange(n_way), k_shot)
    query_patches = torch.randn(n_way * query_shot, 30, 9, 9)
    query_labels = torch.repeat_interleave(torch.arange(n_way), query_shot)

    # Forward pass
    logits, loss, predictions = model(
        support_patches, support_labels, query_patches, query_labels
    )

    print(f"Support shape: {support_patches.shape}")
    print(f"Query shape: {query_patches.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Predictions: {predictions}")

    # Compute accuracy
    accuracy = (predictions == query_labels).float().mean()
    print(f"Accuracy: {accuracy.item():.4f}")
