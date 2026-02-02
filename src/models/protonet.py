from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypicalNetwork(nn.Module):
    """
    Prototypical Network for Few-Shot Classification
    """

    def __init__(
        self,
        backbone: nn.Module,
        distance_metric: str = "euclidean",
    ):
        super().__init__()
        self.backbone = backbone
        self.distance_metric = distance_metric.lower()

        if self.distance_metric not in ["euclidean", "cosine"]:
            raise ValueError(f"Unknown distance metric: {distance_metric}")

        print(f"Initialized ProtoNet with {self.distance_metric} distance.")

    def compute_prototypes(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        n_way: int,
    ) -> torch.Tensor:
        prototypes = []
        for class_idx in range(n_way):
            class_mask = support_labels == class_idx
            class_features = support_features[class_mask]
            # Handle case where a class might be missing in a bad batch
            if class_features.size(0) == 0:
                prototype = torch.zeros(support_features.size(1)).to(
                    support_features.device
                )
            else:
                prototype = class_features.mean(dim=0)
            prototypes.append(prototype)
        return torch.stack(prototypes)

    def compute_distances(
        self,
        query_features: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        if self.distance_metric == "euclidean":
            # ||q - p||^2 = ||q||^2 + ||p||^2 - 2qp
            # More memory efficient implementation
            n_query = query_features.size(0)
            n_proto = prototypes.size(0)

            # (n_query, n_proto)
            dists = torch.cdist(query_features, prototypes, p=2)
            distances = dists**2

        elif self.distance_metric == "cosine":
            # Normalize features
            query_norm = F.normalize(query_features, p=2, dim=1)
            proto_norm = F.normalize(prototypes, p=2, dim=1)

            # Cosine similarity
            similarity = torch.mm(query_norm, proto_norm.t())

            # Convert to 'distance' (1 - similarity or negative similarity)
            # We use negative similarity so that argmax(logits) works same as argmin(distance)
            distances = 1.0 - similarity

        return distances

    def forward(
        self,
        support_patches: torch.Tensor,
        support_labels: torch.Tensor,
        query_patches: torch.Tensor,
        query_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        support_features = self.backbone(support_patches)
        query_features = self.backbone(query_patches)

        n_way = len(torch.unique(support_labels))
        prototypes = self.compute_prototypes(support_features, support_labels, n_way)
        distances = self.compute_distances(query_features, prototypes)
        logits = -distances
        predictions = torch.argmax(logits, dim=1)

        loss = None
        if query_labels is not None:
            loss = F.cross_entropy(logits, query_labels)

        return logits, loss, predictions


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for Spectral Attention"""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C) - Features are already global pooled in backbone
        y = self.fc1(x)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y


class PrototypicalNetworkWithAttention(PrototypicalNetwork):
    """
    Enhanced Prototypical Network with:
    1. Spectral Attention (SE Block) on extracted features.
    2. Self-Attention on support features to refine prototypes.
    """

    def __init__(
        self,
        backbone: nn.Module,
        d_model: int,
        n_heads: int = 4,
        distance_metric: str = "euclidean",
    ):
        super().__init__(backbone, distance_metric)

        # 1. Spectral Attention (Feature Refinement)
        self.se_block = SEBlock(d_model)

        # 2. Self-Attention for Support Set
        self.support_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(
        self,
        support_patches: torch.Tensor,
        support_labels: torch.Tensor,
        query_patches: torch.Tensor,
        query_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Extract Features
        support_features = self.backbone(support_patches)  # (N_sup, d)
        query_features = self.backbone(query_patches)  # (N_qry, d)

        # 2. Apply Spectral Attention (Refine individual features)
        support_features = self.se_block(support_features)
        query_features = self.se_block(query_features)

        # 3. Apply Self-Attention to Support Set (Contextualize)
        # Reshape for Attention: (1, N_sup, d) treating the set as a sequence
        sup_feat_seq = support_features.unsqueeze(0)

        attn_out, _ = self.support_attention(sup_feat_seq, sup_feat_seq, sup_feat_seq)
        # Residual + Norm
        support_features = self.ln(support_features + attn_out.squeeze(0))

        # 4. Standard ProtoNet Flow
        n_way = len(torch.unique(support_labels))
        prototypes = self.compute_prototypes(support_features, support_labels, n_way)
        distances = self.compute_distances(query_features, prototypes)
        logits = -distances
        predictions = torch.argmax(logits, dim=1)

        loss = None
        if query_labels is not None:
            loss = F.cross_entropy(logits, query_labels)

        return logits, loss, predictions
