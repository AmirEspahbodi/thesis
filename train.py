"""
Main Training Script for Few-Shot HSI Classification

This script supports both in-domain and cross-domain strategies:
- In-Domain: Train/Test on same dataset with disjoint pixel sets
- Cross-Domain: Train on source dataset, test on target dataset
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from datamodules import (
    HSIDataset,
    create_data_splits,
    FewShotSampler,
    collate_few_shot_batch,
)
from models import Simple3DCNN, PrototypicalNetwork, count_parameters
from engine import FewShotTrainer, FewShotEvaluator


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

    Returns:
        train_loader, val_loader, test_loader, n_classes
    """
    print("=" * 60)
    print("Creating IN-DOMAIN Dataloaders")
    print("=" * 60)

    # Load full dataset
    full_dataset = HSIDataset(
        data_root=cfg.paths.data_root,
        file_name=cfg.dataset.file_name,
        gt_name=cfg.dataset.gt_name,
        image_key=cfg.dataset.image_key,
        gt_key=cfg.dataset.gt_key,
        patch_size=cfg.data.patch_size,
        target_bands=cfg.data.target_bands,
        ignored_labels=cfg.dataset.ignored_labels,
    )

    n_classes = full_dataset.get_num_classes()
    print(f"Dataset: {cfg.dataset.file_name}")
    print(f"Number of classes: {n_classes}")
    print(f"Total samples: {len(full_dataset)}")

    # Split into train/val/test
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

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_few_shot_batch,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=collate_few_shot_batch,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        collate_fn=collate_few_shot_batch,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, n_classes


def create_cross_domain_dataloaders(cfg: DictConfig):
    """
    Create dataloaders for cross-domain strategy

    Returns:
        train_loader, val_loader, test_loader, n_classes_source, n_classes_target
    """
    print("=" * 60)
    print("Creating CROSS-DOMAIN Dataloaders")
    print("=" * 60)

    # Load source dataset (for training)
    source_cfg = cfg.source_dataset
    source_dataset = HSIDataset(
        data_root=cfg.paths.data_root,
        file_name=source_cfg.file_name,
        gt_name=source_cfg.gt_name,
        image_key=source_cfg.image_key,
        gt_key=source_cfg.gt_key,
        patch_size=cfg.data.patch_size,
        target_bands=cfg.data.target_bands,
        ignored_labels=source_cfg.ignored_labels,
    )

    n_classes_source = source_dataset.get_num_classes()
    print(f"Source Dataset: {source_cfg.file_name}")
    print(f"Source classes: {n_classes_source}")
    print(f"Source samples: {len(source_dataset)}")

    # Split source for train/val
    train_dataset, val_dataset, _ = create_data_splits(
        source_dataset,
        train_ratio=0.8,  # Use more data for training in cross-domain
        val_ratio=0.2,
        seed=cfg.seed,
    )

    # Load target dataset (for testing)
    target_cfg = cfg.target_dataset
    target_dataset = HSIDataset(
        data_root=cfg.paths.data_root,
        file_name=target_cfg.file_name,
        gt_name=target_cfg.gt_name,
        image_key=target_cfg.image_key,
        gt_key=target_cfg.gt_key,
        patch_size=cfg.data.patch_size,
        target_bands=cfg.data.target_bands,
        ignored_labels=target_cfg.ignored_labels,
    )

    n_classes_target = target_dataset.get_num_classes()
    print(f"Target Dataset: {target_cfg.file_name}")
    print(f"Target classes: {n_classes_target}")
    print(f"Target samples: {len(target_dataset)}")

    # Calculate episodes
    n_train_episodes = max(
        100, len(train_dataset) // (cfg.few_shot.n_way * cfg.few_shot.k_shot)
    )
    n_val_episodes = max(
        50, len(val_dataset) // (cfg.few_shot.n_way * cfg.few_shot.k_shot)
    )
    n_test_episodes = max(
        100, len(target_dataset) // (cfg.few_shot.n_way * cfg.few_shot.k_shot)
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
        target_dataset,
        n_way=cfg.few_shot.n_way,
        k_shot=cfg.few_shot.k_shot,
        query_shot=cfg.few_shot.query_shot,
        n_episodes=n_test_episodes,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_few_shot_batch,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=collate_few_shot_batch,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        target_dataset,
        batch_sampler=test_sampler,
        collate_fn=collate_few_shot_batch,
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
    main()
