"""
Training and Evaluation Engine

Implements training and evaluation loops for few-shot learning.
Supports both in-domain and cross-domain strategies.
"""

import os
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.metrics import (
    AccuracyCalculator,
    RunningAverage,
    compute_episode_accuracy,
)


class FewShotTrainer:
    """
    Trainer for Few-Shot Learning

    Handles:
    - Training loop with episodic batches
    - Validation
    - Model checkpointing
    - Logging

    Args:
        model: Prototypical Network
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler (optional)
        device: Device to run on ('cuda' or 'cpu')
        n_way: Number of classes per episode
        k_shot: Number of support samples per class
        query_shot: Number of query samples per class
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        n_way: int = 5,
        k_shot: int = 5,
        query_shot: int = 15,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_shot = query_shot

        # Move model to device
        self.model.to(device)

        # Metrics tracking
        self.train_loss = RunningAverage()
        self.train_acc = RunningAverage()

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        log_interval: int = 10,
    ) -> Dict[str, float]:
        """
        Train for one epoch

        Args:
            dataloader: DataLoader with episodic sampling
            epoch: Current epoch number
            log_interval: Log every N episodes

        Returns:
            Dictionary with average loss and accuracy
        """
        self.model.train()
        self.train_loss.reset()
        self.train_acc.reset()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

        for episode_idx, (support_x, support_y, query_x, query_y) in enumerate(pbar):
            # Move to device
            support_x = support_x.to(self.device)
            support_y = support_y.to(self.device)
            query_x = query_x.to(self.device)
            query_y = query_y.to(self.device)

            # Remap labels to [0, n_way) for each episode
            unique_labels = torch.unique(support_y)
            label_map = {label.item(): idx for idx, label in enumerate(unique_labels)}

            support_y_mapped = torch.tensor(
                [label_map[label.item()] for label in support_y], device=self.device
            )
            query_y_mapped = torch.tensor(
                [label_map[label.item()] for label in query_y], device=self.device
            )

            # Forward pass
            logits, loss, predictions = self.model(
                support_x, support_y_mapped, query_x, query_y_mapped
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Compute accuracy
            accuracy = compute_episode_accuracy(predictions, query_y_mapped)

            # Update metrics
            self.train_loss.update(loss.item())
            self.train_acc.update(accuracy)

            # Update progress bar
            if episode_idx % log_interval == 0:
                pbar.set_postfix(
                    {
                        "loss": f"{self.train_loss.get():.4f}",
                        "acc": f"{self.train_acc.get():.4f}",
                    }
                )

        return {"loss": self.train_loss.get(), "accuracy": self.train_acc.get()}

    def validate(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """
        Validate the model

        Args:
            dataloader: Validation dataloader
            epoch: Current epoch number

        Returns:
            Dictionary with average loss and accuracy
        """
        self.model.eval()
        val_loss = RunningAverage()
        val_acc = RunningAverage()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")

        with torch.no_grad():
            for support_x, support_y, query_x, query_y in pbar:
                # Move to device
                support_x = support_x.to(self.device)
                support_y = support_y.to(self.device)
                query_x = query_x.to(self.device)
                query_y = query_y.to(self.device)

                # Remap labels
                unique_labels = torch.unique(support_y)
                label_map = {
                    label.item(): idx for idx, label in enumerate(unique_labels)
                }

                support_y_mapped = torch.tensor(
                    [label_map[label.item()] for label in support_y], device=self.device
                )
                query_y_mapped = torch.tensor(
                    [label_map[label.item()] for label in query_y], device=self.device
                )

                # Forward pass
                logits, loss, predictions = self.model(
                    support_x, support_y_mapped, query_x, query_y_mapped
                )

                # Compute accuracy
                accuracy = compute_episode_accuracy(predictions, query_y_mapped)

                # Update metrics
                val_loss.update(loss.item())
                val_acc.update(accuracy)

                pbar.set_postfix(
                    {"loss": f"{val_loss.get():.4f}", "acc": f"{val_acc.get():.4f}"}
                )

        return {"loss": val_loss.get(), "accuracy": val_acc.get()}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        checkpoint_dir: str,
        log_interval: int = 10,
        patience: int = 15,
        min_delta: float = 0.001,
    ) -> Dict[str, list]:
        """
        Full training loop with early stopping

        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints
            log_interval: Log every N episodes
            patience: Early stopping patience
            min_delta: Minimum improvement for early stopping

        Returns:
            Dictionary with training history
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            # Training
            train_metrics = self.train_epoch(train_loader, epoch, log_interval)
            history["train_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["accuracy"])

            # Validation
            val_metrics = self.validate(val_loader, epoch)
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()

            # Print epoch summary
            print(f"\nEpoch {epoch}/{epochs}")
            print(
                f"Train - Loss: {train_metrics['loss']:.4f}, "
                f"Acc: {train_metrics['accuracy']:.4f}"
            )
            print(
                f"Val   - Loss: {val_metrics['loss']:.4f}, "
                f"Acc: {val_metrics['accuracy']:.4f}"
            )

            # Save best model
            if val_metrics["accuracy"] > best_val_acc + min_delta:
                best_val_acc = val_metrics["accuracy"]
                patience_counter = 0

                checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_acc": best_val_acc,
                    },
                    checkpoint_path,
                )

                print(f"✓ Saved best model (val_acc: {best_val_acc:.4f})")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break

            print("-" * 60)

        print(f"\nTraining completed. Best val accuracy: {best_val_acc:.4f}")

        return history


class FewShotEvaluator:
    """
    Evaluator for Few-Shot Learning

    Performs comprehensive evaluation and computes detailed metrics.

    Args:
        model: Prototypical Network
        device: Device to run on
        n_way: Number of classes per episode
        k_shot: Number of support samples per class
        query_shot: Number of query samples per class
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        n_way: int = 5,
        k_shot: int = 5,
        query_shot: int = 15,
    ):
        self.model = model
        self.device = device
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_shot = query_shot

        self.model.to(device)

    def evaluate(
        self,
        dataloader: DataLoader,
        n_classes: int,
    ) -> Dict[str, float]:
        """
        Evaluate model and compute detailed metrics

        Args:
            dataloader: Test dataloader
            n_classes: Total number of classes in dataset

        Returns:
            Dictionary with OA, AA, Kappa, and per-class accuracies
        """
        self.model.eval()

        # Accumulate predictions
        calc = AccuracyCalculator(n_classes)

        pbar = tqdm(dataloader, desc="Evaluating")

        with torch.no_grad():
            for support_x, support_y, query_x, query_y in pbar:
                # Move to device
                support_x = support_x.to(self.device)
                support_y = support_y.to(self.device)
                query_x = query_x.to(self.device)
                query_y = query_y.to(self.device)

                # Remap labels for this episode
                unique_labels = torch.unique(support_y)
                label_map = {
                    label.item(): idx for idx, label in enumerate(unique_labels)
                }

                # Create reverse map for predictions
                reverse_map = {idx: label.item() for label, idx in label_map.items()}

                support_y_mapped = torch.tensor(
                    [label_map[label.item()] for label in support_y], device=self.device
                )
                query_y_mapped = torch.tensor(
                    [label_map[label.item()] for label in query_y], device=self.device
                )

                # Forward pass
                _, _, predictions_mapped = self.model(
                    support_x, support_y_mapped, query_x, query_y_mapped
                )

                # Map predictions back to original label space
                predictions = torch.tensor(
                    [reverse_map[pred.item()] for pred in predictions_mapped],
                    device=self.device,
                )

                # Update calculator with original labels
                calc.update(predictions, query_y)

        # Compute final metrics
        metrics = calc.compute()

        # Print summary
        print("\n" + calc.get_summary())

        return metrics

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Checkpoint val_acc: {checkpoint.get('val_acc', 'N/A')}")
