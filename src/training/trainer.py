"""Training engine for few-shot hyperspectral image classification.

This module implements the training and evaluation logic for episodic
few-shot learning with Prototypical Networks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Tuple
import numpy as np
from pathlib import Path

from ..models import PrototypicalNetwork
from ..data import EpisodeSampler
from ..utils import (
    ClassificationMetrics,
    AverageMeter,
    save_checkpoint,
    ExperimentTracker
)


class FewShotTrainer:
    """Trainer for few-shot learning with episodic sampling.
    
    This class manages the training loop, evaluation, and checkpoint saving
    for few-shot classification models.
    """
    
    def __init__(
        self,
        model: PrototypicalNetwork,
        train_sampler: EpisodeSampler,
        test_sampler: EpisodeSampler,
        optimizer: optim.Optimizer,
        device: str = 'cuda',
        checkpoint_dir: str = 'checkpoints',
        log_interval: int = 50
    ):
        """Initialize the trainer.
        
        Args:
            model: Prototypical Network model.
            train_sampler: Episodic sampler for training.
            test_sampler: Episodic sampler for testing.
            optimizer: Optimizer for training.
            device: Device to use for training.
            checkpoint_dir: Directory to save checkpoints.
            log_interval: Interval for logging (in episodes).
        """
        self.model = model.to(device)
        self.train_sampler = train_sampler
        self.test_sampler = test_sampler
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Best model tracking
        self.best_accuracy = 0.0
        self.best_epoch = 0
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch (all episodes).
        
        Args:
            epoch: Current epoch number.
            
        Returns:
            Tuple of (average_loss, average_accuracy).
        """
        self.model.train()
        
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        
        # Iterate through episodes
        for episode_idx, episode in enumerate(self.train_sampler):
            # Move data to device
            support_data = episode['support_data'].to(self.device)
            support_labels = episode['support_labels'].to(self.device)
            query_data = episode['query_data'].to(self.device)
            query_labels = episode['query_labels'].to(self.device)
            
            # Forward pass
            logits, _ = self.model(
                support_data,
                support_labels,
                query_data,
                n_way=self.train_sampler.n_way
            )
            
            # Calculate loss
            loss = self.criterion(logits, query_labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == query_labels).float().mean()
            
            # Update meters
            loss_meter.update(loss.item(), query_data.size(0))
            acc_meter.update(accuracy.item(), query_data.size(0))
            
            # Logging
            if (episode_idx + 1) % self.log_interval == 0:
                print(
                    f"Epoch [{epoch}] Episode [{episode_idx + 1}/{len(self.train_sampler)}] "
                    f"Loss: {loss_meter.avg:.4f} | Acc: {acc_meter.avg:.4f}"
                )
        
        return loss_meter.avg, acc_meter.avg
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on test episodes.
        
        Returns:
            Dictionary containing evaluation metrics.
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        
        loss_meter = AverageMeter()
        
        # Iterate through test episodes
        for episode_idx, episode in enumerate(self.test_sampler):
            # Move data to device
            support_data = episode['support_data'].to(self.device)
            support_labels = episode['support_labels'].to(self.device)
            query_data = episode['query_data'].to(self.device)
            query_labels = episode['query_labels'].to(self.device)
            
            # Forward pass
            logits, _ = self.model(
                support_data,
                support_labels,
                query_data,
                n_way=self.test_sampler.n_way
            )
            
            # Calculate loss
            loss = self.criterion(logits, query_labels)
            loss_meter.update(loss.item(), query_data.size(0))
            
            # Get predictions
            predictions = torch.argmax(logits, dim=1)
            
            # Collect predictions and labels
            all_predictions.append(predictions.cpu())
            all_labels.append(query_labels.cpu())
        
        # Concatenate all predictions and labels
        all_predictions = torch.cat(all_predictions).numpy()
        all_labels = torch.cat(all_labels).numpy()
        
        # Calculate metrics
        metrics = ClassificationMetrics.compute_all_metrics(
            all_labels,
            all_predictions,
            verbose=True
        )
        
        metrics['loss'] = loss_meter.avg
        
        return metrics
    
    def train(
        self,
        num_epochs: int,
        save_best_only: bool = True
    ) -> Dict[str, List[float]]:
        """Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train.
            save_best_only: Whether to save only the best model.
            
        Returns:
            Dictionary containing training history.
        """
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': []
        }
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 60)
            
            # Train for one epoch
            train_loss, train_acc = self.train_epoch(epoch)
            
            print(f"\nTraining Results:")
            print(f"  Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")
            
            # Evaluate
            print(f"\nEvaluating on test episodes...")
            test_metrics = self.evaluate()
            
            # Store history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_metrics['loss'])
            history['test_acc'].append(test_metrics['overall_accuracy'])
            
            # Save checkpoint
            is_best = test_metrics['overall_accuracy'] > self.best_accuracy
            
            if is_best:
                self.best_accuracy = test_metrics['overall_accuracy']
                self.best_epoch = epoch
            
            if not save_best_only or is_best:
                checkpoint_path = self.checkpoint_dir / f"model_epoch_{epoch}.pt"
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    train_loss,
                    str(checkpoint_path)
                )
                
                if is_best:
                    best_path = self.checkpoint_dir / "best_model.pt"
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        epoch,
                        train_loss,
                        str(best_path)
                    )
                    print(f"New best model! Accuracy: {self.best_accuracy:.4f}")
        
        print("\n" + "="*60)
        print("Training Completed")
        print("="*60)
        print(f"Best Accuracy: {self.best_accuracy:.4f} at epoch {self.best_epoch}")
        
        return history
    
    @torch.no_grad()
    def test_on_full_dataset(
        self,
        test_dataset,
        batch_size: int = 64
    ) -> Dict[str, float]:
        """Test on full dataset (not episodic).
        
        This method tests the model on all samples in the test dataset
        by using the mean of all training class prototypes.
        
        Args:
            test_dataset: Test dataset.
            batch_size: Batch size for testing.
            
        Returns:
            Dictionary containing test metrics.
        """
        self.model.eval()
        
        # Create dataloader
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        all_predictions = []
        all_labels = []
        
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(self.device)
            
            # Extract features
            features = self.model.extract_features(batch_data)
            
            # Note: This is a simplified version
            # In practice, you'd need to compute prototypes from support set
            # For now, we just extract features
            
            all_predictions.append(features.cpu())
            all_labels.append(batch_labels)
        
        # This would need proper implementation for full dataset testing
        print("Full dataset testing requires pre-computed prototypes.")
        print("Use episodic evaluation instead.")
        
        return {}
