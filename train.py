import os
import random
import sys
from functools import partial

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from datamodules import (
    FewShotSampler,
    HSIDataset,
    collate_few_shot_batch,
    create_collate_fn,
    create_data_splits,
)
from datamodules.hsi_dataset import fit_pca_on_indices
from engine import FewShotEvaluator, FewShotTrainer
from models import PrototypicalNetwork, Simple3DCNN, count_parameters


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_in_domain_dataloaders(cfg: DictConfig):
    """
    Create dataloaders for in-domain strategy

    FIX #2: Uses create_data_splits which prevents PCA leakage

    Returns:
        train_loader, val_loader, test_loader, n_classes
    """
    print("=" * 60)
    print("Creating IN-DOMAIN Dataloaders")
    print("=" * 60)

    # Load full dataset WITHOUT PCA
    full_dataset = HSIDataset(
        data_root=cfg.paths.data_root,
        file_name=cfg.dataset.file_name,
        gt_name=cfg.dataset.gt_name,
        image_key=cfg.dataset.image_key,
        gt_key=cfg.dataset.gt_key,
        patch_size=cfg.data.patch_size,
        target_bands=cfg.data.target_bands,
        ignored_labels=cfg.dataset.ignored_labels,
        pca_transformer=None,  # Don't apply PCA yet
    )

    n_classes = full_dataset.get_num_classes()
    print(f"Dataset: {cfg.dataset.file_name}")
    print(f"Number of classes: {n_classes}")
    print(f"Total samples: {len(full_dataset)}")

    # Split with leak-free PCA
    train_dataset, val_dataset, test_dataset = create_data_splits(
        full_dataset,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        seed=cfg.seed,
    )

    # Calculate number of episodes
    n_train_episodes = max(
        100, len(train_dataset) // (cfg.few_shot.n_way * cfg.few_shot.k_shot)
    )
    n_val_episodes = max(
        50, len(val_dataset) // (cfg.few_shot.n_way * cfg.few_shot.k_shot)
    )
    n_test_episodes = max(
        100, len(test_dataset) // (cfg.few_shot.n_way * cfg.few_shot.k_shot)
    )

    print(
        f"Episodes - Train: {n_train_episodes}, Val: {n_val_episodes}, Test: {n_test_episodes}"
    )

    # Create samplers
    train_sampler = FewShotSampler(
        train_dataset,
        n_way=cfg.few_shot.n_way,
        k_shot=cfg.few_shot.k_shot,
        query_shot=cfg.few_shot.query_shot,
        n_episodes=n_train_episodes,
    )

    val_sampler = FewShotSampler(
        val_dataset,
        n_way=cfg.few_shot.n_way,
        k_shot=cfg.few_shot.k_shot,
        query_shot=cfg.few_shot.query_shot,
        n_episodes=n_val_episodes,
    )

    test_sampler = FewShotSampler(
        test_dataset,
        n_way=cfg.few_shot.n_way,
        k_shot=cfg.few_shot.k_shot,
        query_shot=cfg.few_shot.query_shot,
        n_episodes=n_test_episodes,
    )

    # Create collate function with injected parameters (FIX #1)
    collate_fn = partial(
        collate_few_shot_batch,
        n_way=cfg.few_shot.n_way,
        k_shot=cfg.few_shot.k_shot,
        query_shot=cfg.few_shot.query_shot,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, n_classes


def create_cross_domain_dataloaders(cfg: DictConfig):
    """
    Create dataloaders for cross-domain strategy with COMPATIBLE PCA.

    FIX #3: Ensures source and target datasets use the SAME PCA basis
    for compatible feature spaces.

    Returns:
        train_loader, val_loader, test_loader, n_classes_source, n_classes_target
    """
    print("=" * 60)
    print("Creating CROSS-DOMAIN Dataloaders")
    print("=" * 60)

    # === Load Source Dataset (RAW - no PCA yet) ===
    source_cfg = cfg.source_dataset
    print(f"\nLoading SOURCE dataset: {source_cfg.file_name}")

    source_dataset_raw = HSIDataset(
        data_root=cfg.paths.data_root,
        file_name=source_cfg.file_name,
        gt_name=source_cfg.gt_name,
        image_key=source_cfg.image_key,
        gt_key=source_cfg.gt_key,
        patch_size=cfg.data.patch_size,
        target_bands=cfg.data.target_bands,
        ignored_labels=source_cfg.ignored_labels,
        pca_transformer=None,  # Will fit below
    )

    n_classes_source = source_dataset_raw.get_num_classes()
    print(f"  Source classes: {n_classes_source}")
    print(f"  Source samples: {len(source_dataset_raw)}")
    print(f"  Source bands: {source_dataset_raw.image_raw.shape[2]}")

    # === FIX #3: Fit PCA on SOURCE dataset ===
    print(f"\nFIX #3: Fitting PCA on SOURCE dataset for cross-domain compatibility")

    # Get all source indices for PCA fitting
    source_indices = source_dataset_raw.valid_indices

    # Fit PCA on source
    scaler_source, pca_source = fit_pca_on_indices(
        image=source_dataset_raw.image_raw,
        indices=source_indices,
        gt_shape=source_dataset_raw.gt.shape,
        target_bands=cfg.data.target_bands,
    )

    pca_transformer = (scaler_source, pca_source)

    # Apply PCA to source dataset
    H_src, W_src, C_src = source_dataset_raw.image_raw.shape
    source_2d = source_dataset_raw.image_raw.reshape(-1, C_src)
    source_2d = scaler_source.transform(source_2d)
    source_pca = pca_source.transform(source_2d)
    source_transformed = source_pca.reshape(H_src, W_src, cfg.data.target_bands).astype(
        np.float32
    )

    # Update source dataset with transformed image
    source_dataset_raw.image = source_dataset_raw._pad_image(source_transformed)
    source_dataset_raw.pca_transformer = pca_transformer

    # Split source for train/val (using already-transformed data)
    # We need to split WITHOUT refitting PCA
    print(f"\nSplitting source dataset (80% train, 20% val)...")

    # Manual split since source is already PCA-transformed
    np.random.seed(cfg.seed)
    source_class_indices = {}
    for idx in source_dataset_raw.valid_indices:
        x, y = (
            idx // source_dataset_raw.gt.shape[1],
            idx % source_dataset_raw.gt.shape[1],
        )
        class_idx = source_dataset_raw.label_map[source_dataset_raw.gt[x, y]]
        source_class_indices.setdefault(class_idx, []).append(idx)

    train_indices = []
    val_indices = []

    for class_idx, indices in source_class_indices.items():
        indices = np.array(indices)
        np.random.shuffle(indices)
        n_train = int(len(indices) * 0.8)
        train_indices.extend(indices[:n_train].tolist())
        val_indices.extend(indices[n_train:].tolist())

    def _wrap_source(idx_list):
        ds = HSIDataset(
            source_dataset_raw.data_root,
            None,
            None,
            None,
            None,
            source_dataset_raw.patch_size,
            source_dataset_raw.target_bands,
            source_dataset_raw.ignored_labels,
            idx_list,
            pca_transformer=pca_transformer,
        )
        ds.image_raw = source_dataset_raw.image_raw
        ds.image = source_dataset_raw.image
        ds.gt = source_dataset_raw.gt
        ds.label_map = source_dataset_raw.label_map
        ds.valid_indices = source_dataset_raw.valid_indices
        return ds

    train_dataset = _wrap_source(train_indices)
    val_dataset = _wrap_source(val_indices)

    # === Load Target Dataset and Apply SAME PCA ===
    target_cfg = cfg.target_dataset
    print(f"\nLoading TARGET dataset: {target_cfg.file_name}")

    # Load raw target
    target_dataset_raw = HSIDataset(
        data_root=cfg.paths.data_root,
        file_name=target_cfg.file_name,
        gt_name=target_cfg.gt_name,
        image_key=target_cfg.image_key,
        gt_key=target_cfg.gt_key,
        patch_size=cfg.data.patch_size,
        target_bands=cfg.data.target_bands,
        ignored_labels=target_cfg.ignored_labels,
        pca_transformer=None,
    )

    n_classes_target = target_dataset_raw.get_num_classes()
    print(f"  Target classes: {n_classes_target}")
    print(f"  Target samples: {len(target_dataset_raw)}")
    print(f"  Target bands: {target_dataset_raw.image_raw.shape[2]}")

    # Check band compatibility
    H_tgt, W_tgt, C_tgt = target_dataset_raw.image_raw.shape

    if C_tgt != C_src:
        print(f"\n  ⚠️  WARNING: Source has {C_src} bands, Target has {C_tgt} bands")
        print(f"     Interpolating target spectral bands to match source...")

        # Use PyTorch for interpolation (H, W, C) -> (H*W, 1, C)
        img_tensor = torch.from_numpy(target_dataset_raw.image_raw).float()
        img_tensor = img_tensor.reshape(
            -1, 1, C_tgt
        )  # Treat as (Batch, Channel, Length)

        # Interpolate to C_src
        img_interpolated = F.interpolate(
            img_tensor, size=C_src, mode="linear", align_corners=False
        )

        # Reshape back to (H, W, C_src)
        target_dataset_raw.image_raw = img_interpolated.reshape(
            H_tgt, W_tgt, C_src
        ).numpy()

        print(
            f"  ✓ Interpolated target bands from {C_tgt} to {C_src} to match source PCA."
        )

        # Update C_tgt to match new dimension so subsequent code uses correct shape
        C_tgt = C_src

    # Apply SOURCE PCA to target
    print(f"\n  Applying SOURCE PCA to target dataset...")
    target_2d = target_dataset_raw.image_raw.reshape(-1, C_tgt)
    target_2d = scaler_source.transform(target_2d)  # Use SOURCE scaler
    target_pca = pca_source.transform(target_2d)  # Use SOURCE PCA
    target_transformed = target_pca.reshape(H_tgt, W_tgt, cfg.data.target_bands).astype(
        np.float32
    )

    target_dataset_raw.image = target_dataset_raw._pad_image(target_transformed)
    target_dataset_raw.pca_transformer = pca_transformer  # Same as source

    print(f"  ✓ Source and target now share SAME PCA basis (compatible feature space)")

    # Calculate episodes
    n_train_episodes = max(
        100, len(train_dataset) // (cfg.few_shot.n_way * cfg.few_shot.k_shot)
    )
    n_val_episodes = max(
        50, len(val_dataset) // (cfg.few_shot.n_way * cfg.few_shot.k_shot)
    )
    n_test_episodes = max(
        100, len(target_dataset_raw) // (cfg.few_shot.n_way * cfg.few_shot.k_shot)
    )

    print(
        f"\nEpisodes - Train: {n_train_episodes}, Val: {n_val_episodes}, Test: {n_test_episodes}"
    )

    # Create samplers
    train_sampler = FewShotSampler(
        train_dataset,
        n_way=cfg.few_shot.n_way,
        k_shot=cfg.few_shot.k_shot,
        query_shot=cfg.few_shot.query_shot,
        n_episodes=n_train_episodes,
    )

    val_sampler = FewShotSampler(
        val_dataset,
        n_way=cfg.few_shot.n_way,
        k_shot=cfg.few_shot.k_shot,
        query_shot=cfg.few_shot.query_shot,
        n_episodes=n_val_episodes,
    )

    test_sampler = FewShotSampler(
        target_dataset_raw,
        n_way=cfg.few_shot.n_way,
        k_shot=cfg.few_shot.k_shot,
        query_shot=cfg.few_shot.query_shot,
        n_episodes=n_test_episodes,
    )

    # Create dataloaders with FIXED collate function
    collate_fn = partial(
        collate_few_shot_batch,
        n_way=cfg.few_shot.n_way,
        k_shot=cfg.few_shot.k_shot,
        query_shot=cfg.few_shot.query_shot,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        target_dataset_raw,
        batch_sampler=test_sampler,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, n_classes_source, n_classes_target


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main training and evaluation function"""

    # Print configuration
    print("\n" + "=" * 60)
    print("Configuration")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))

    # Set seed
    set_seed(cfg.seed)

    # Set device
    device = cfg.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create dataloaders based on strategy
    if cfg.strategy == "in_domain":
        train_loader, val_loader, test_loader, n_classes = create_in_domain_dataloaders(
            cfg
        )
        n_test_classes = n_classes
    elif cfg.strategy == "cross_domain":
        train_loader, val_loader, test_loader, n_classes, n_test_classes = (
            create_cross_domain_dataloaders(cfg)
        )
    else:
        raise ValueError(f"Unknown strategy: {cfg.strategy}")

    # Create model
    print("\n" + "=" * 60)
    print("Creating Model")
    print("=" * 60)

    backbone = Simple3DCNN(
        input_channels=cfg.model.backbone.input_channels,
        spectral_size=cfg.data.target_bands,
        spatial_size=cfg.data.patch_size,
        d_model=cfg.model.backbone.d_model,
    )

    model = PrototypicalNetwork(backbone)

    print(f"Backbone: {cfg.model.backbone.name}")
    print(f"Feature dimension: {cfg.model.backbone.d_model}")
    print(f"Total parameters: {count_parameters(model):,}")

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    # Create scheduler
    scheduler = None
    if cfg.training.scheduler.type == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.training.scheduler.step_size,
            gamma=cfg.training.scheduler.gamma,
        )

    # Create trainer
    trainer = FewShotTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        n_way=cfg.few_shot.n_way,
        k_shot=cfg.few_shot.k_shot,
        query_shot=cfg.few_shot.query_shot,
    )

    # Train
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.training.epochs,
        checkpoint_dir=cfg.paths.checkpoint_dir,
        log_interval=cfg.logging.log_interval,
        patience=cfg.training.early_stopping.patience,
        min_delta=cfg.training.early_stopping.min_delta,
    )

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)

    evaluator = FewShotEvaluator(
        model=model,
        device=device,
        n_way=cfg.few_shot.n_way,
        k_shot=cfg.few_shot.k_shot,
        query_shot=cfg.few_shot.query_shot,
    )

    # Load best model
    best_model_path = os.path.join(cfg.paths.checkpoint_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        evaluator.load_checkpoint(best_model_path)

    # Evaluate
    test_metrics = evaluator.evaluate(test_loader, n_test_classes)

    # Save final results
    results_path = os.path.join(cfg.paths.output_dir, "results.txt")
    os.makedirs(cfg.paths.output_dir, exist_ok=True)

    with open(results_path, "w") as f:
        f.write(f"Few-Shot HSI Classification Results\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Strategy: {cfg.strategy}\n")
        f.write(
            f"Dataset: {cfg.dataset.file_name if cfg.strategy == 'in_domain' else f'{cfg.source_dataset.file_name} -> {cfg.target_dataset.file_name}'}\n"
        )
        f.write(f"n-way: {cfg.few_shot.n_way}, k-shot: {cfg.few_shot.k_shot}\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(
            f"Overall Accuracy (OA): {test_metrics['OA']:.4f} ({test_metrics['OA'] * 100:.2f}%)\n"
        )
        f.write(
            f"Average Accuracy (AA): {test_metrics['AA']:.4f} ({test_metrics['AA'] * 100:.2f}%)\n"
        )
        f.write(f"Kappa Coefficient:     {test_metrics['Kappa']:.4f}\n")

    print(f"\nResults saved to {results_path}")
    print("\nTraining and evaluation completed!")


if __name__ == "__main__":
    import argparse
    import sys

    # FIX: Python 3.14 breaks Hydra's help argument.
    # We force all help arguments to be strings before argparse sees them.
    if sys.version_info >= (3, 14):
        _orig_add_argument = argparse.ArgumentParser.add_argument

        def _fixed_add_argument(self, *args, **kwargs):
            if "help" in kwargs and not isinstance(kwargs["help"], str):
                kwargs["help"] = str(kwargs["help"])
            return _orig_add_argument(self, *args, **kwargs)

        argparse.ArgumentParser.add_argument = _fixed_add_argument
    main()
