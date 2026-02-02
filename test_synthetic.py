"""
Quick Test Script with Synthetic Data

This script runs a minimal training loop with synthetic data to verify
that the entire pipeline works correctly without needing real datasets.
"""

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from models import Simple3DCNN, PrototypicalNetwork
from datamodules.samplers import collate_few_shot_batch
from engine import FewShotTrainer


class SyntheticHSIDataset(Dataset):
    """Synthetic HSI dataset for testing"""

    def __init__(self, n_samples=500, n_classes=10, n_bands=30, patch_size=9):
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.n_bands = n_bands
        self.patch_size = patch_size

        # Generate random data
        np.random.seed(42)
        self.data = []
        self.labels = []

        for i in range(n_samples):
            # Create synthetic patch
            patch = np.random.randn(n_bands, patch_size, patch_size).astype(np.float32)
            label = i % n_classes  # Distribute evenly across classes

            self.data.append(patch)
            self.labels.append(label)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), self.labels[idx]


def create_synthetic_episode(dataset, n_way=5, k_shot=5, query_shot=15):
    """Create a single synthetic episode"""

    # Select n_way classes
    available_classes = list(range(dataset.n_classes))
    selected_classes = np.random.choice(available_classes, n_way, replace=False)

    # Collect samples for each class
    support_patches = []
    support_labels = []
    query_patches = []
    query_labels = []

    for class_idx, original_class in enumerate(selected_classes):
        # Find samples of this class
        class_samples = [
            i for i, label in enumerate(dataset.labels) if label == original_class
        ]

        # Sample support and query
        selected = np.random.choice(class_samples, k_shot + query_shot, replace=False)
        support_indices = selected[:k_shot]
        query_indices = selected[k_shot:]

        # Collect support
        for idx in support_indices:
            patch, _ = dataset[idx]
            support_patches.append(patch)
            support_labels.append(class_idx)

        # Collect query
        for idx in query_indices:
            patch, _ = dataset[idx]
            query_patches.append(patch)
            query_labels.append(class_idx)

    support_patches = torch.stack(support_patches)
    support_labels = torch.tensor(support_labels)
    query_patches = torch.stack(query_patches)
    query_labels = torch.tensor(query_labels)

    return support_patches, support_labels, query_patches, query_labels


class SyntheticDataLoader:
    """Simple data loader for synthetic episodes"""

    def __init__(self, dataset, n_episodes, n_way=5, k_shot=5, query_shot=15):
        self.dataset = dataset
        self.n_episodes = n_episodes
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_shot = query_shot

    def __iter__(self):
        for _ in range(self.n_episodes):
            yield create_synthetic_episode(
                self.dataset, self.n_way, self.k_shot, self.query_shot
            )

    def __len__(self):
        return self.n_episodes


def main():
    print("=" * 60)
    print("Quick Test with Synthetic Data")
    print("=" * 60)

    # Configuration
    n_classes = 10
    n_way = 5
    k_shot = 5
    query_shot = 15
    n_bands = 30
    patch_size = 9
    d_model = 64
    n_train_episodes = 20
    n_val_episodes = 10
    epochs = 3

    print(f"\nConfiguration:")
    print(f"  Classes: {n_classes}")
    print(f"  n-way: {n_way}, k-shot: {k_shot}, query: {query_shot}")
    print(f"  Spectral bands: {n_bands}, Patch size: {patch_size}")
    print(f"  Feature dimension: {d_model}")
    print(f"  Training episodes: {n_train_episodes}")
    print(f"  Validation episodes: {n_val_episodes}")
    print(f"  Epochs: {epochs}")

    # Create synthetic datasets
    print("\nCreating synthetic datasets...")
    train_dataset = SyntheticHSIDataset(
        n_samples=500, n_classes=n_classes, n_bands=n_bands, patch_size=patch_size
    )
    val_dataset = SyntheticHSIDataset(
        n_samples=200, n_classes=n_classes, n_bands=n_bands, patch_size=patch_size
    )
    print(f"✓ Train dataset: {len(train_dataset)} samples")
    print(f"✓ Val dataset: {len(val_dataset)} samples")

    # Create data loaders
    print("\nCreating data loaders...")
    train_loader = SyntheticDataLoader(
        train_dataset,
        n_episodes=n_train_episodes,
        n_way=n_way,
        k_shot=k_shot,
        query_shot=query_shot,
    )
    val_loader = SyntheticDataLoader(
        val_dataset,
        n_episodes=n_val_episodes,
        n_way=n_way,
        k_shot=k_shot,
        query_shot=query_shot,
    )
    print("✓ Data loaders created")

    # Create model
    print("\nCreating model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    backbone = Simple3DCNN(
        input_channels=1,
        spectral_size=n_bands,
        spatial_size=patch_size,
        d_model=d_model,
    )

    model = PrototypicalNetwork(backbone)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model created with {n_params:,} parameters")

    # Test forward pass
    print("\nTesting forward pass...")
    test_support, test_support_labels, test_query, test_query_labels = next(
        iter(train_loader)
    )
    test_support = test_support.to(device)
    test_support_labels = test_support_labels.to(device)
    test_query = test_query.to(device)
    test_query_labels = test_query_labels.to(device)

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        logits, loss, predictions = model(
            test_support, test_support_labels, test_query, test_query_labels
        )

    accuracy = (predictions == test_query_labels).float().mean()
    print(f"✓ Forward pass successful")
    print(f"  Support shape: {test_support.shape}")
    print(f"  Query shape: {test_query.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Random accuracy: {accuracy.item():.4f}")

    # Create trainer
    print("\nCreating trainer...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    trainer = FewShotTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        n_way=n_way,
        k_shot=k_shot,
        query_shot=query_shot,
    )
    print("✓ Trainer created")

    # Run minimal training
    print("\n" + "=" * 60)
    print("Running minimal training loop...")
    print("=" * 60)

    os.makedirs("test_checkpoints", exist_ok=True)

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        checkpoint_dir="test_checkpoints",
        log_interval=5,
        patience=10,
        min_delta=0.0,
    )

    # Print results
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)

    print("\nFinal metrics:")
    print(f"  Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"  Train Acc:  {history['train_acc'][-1]:.4f}")
    print(f"  Val Loss:   {history['val_loss'][-1]:.4f}")
    print(f"  Val Acc:    {history['val_acc'][-1]:.4f}")

    print("\n✓ All components working correctly!")
    print("\nYou can now run the full training with real data:")
    print("  python train.py dataset=houston13")

    # Clean up
    import shutil

    if os.path.exists("test_checkpoints"):
        shutil.rmtree("test_checkpoints")
        print("\n✓ Test checkpoints cleaned up")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
