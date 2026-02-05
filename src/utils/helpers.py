"""Utility functions for logging, visualization, and helpers.

This module provides logging setup, reproducibility functions,
and other helper utilities.
"""

import torch
import numpy as np
import random
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime


def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility.
    
    Sets seeds for Python's random, NumPy, and PyTorch (both CPU and CUDA).
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to: {seed}")


def setup_logger(
    name: str = "hsi_fewshot",
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """Setup logger for the project.
    
    Args:
        name: Logger name.
        log_file: Optional path to log file.
        level: Logging level.
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if log_file specified
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class ExperimentTracker:
    """Track experiment metrics and save results.
    
    This class manages logging of training/testing metrics and saves
    results to JSON files.
    """
    
    def __init__(self, experiment_name: str, save_dir: str = "results"):
        """Initialize the experiment tracker.
        
        Args:
            experiment_name: Name of the experiment.
            save_dir: Directory to save results.
        """
        self.experiment_name = experiment_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_metrics: Dict[str, list] = {
            'epoch': [],
            'loss': [],
            'accuracy': []
        }
        
        self.test_metrics: Dict[str, Any] = {}
        
        self.start_time = datetime.now()
    
    def log_train_metrics(
        self,
        epoch: int,
        loss: float,
        accuracy: float
    ):
        """Log training metrics for an epoch.
        
        Args:
            epoch: Epoch number.
            loss: Training loss.
            accuracy: Training accuracy.
        """
        self.train_metrics['epoch'].append(epoch)
        self.train_metrics['loss'].append(float(loss))
        self.train_metrics['accuracy'].append(float(accuracy))
    
    def log_test_metrics(self, metrics: Dict[str, float]):
        """Log testing metrics.
        
        Args:
            metrics: Dictionary of test metrics.
        """
        self.test_metrics = {k: float(v) for k, v in metrics.items()}
    
    def save_results(self):
        """Save all results to JSON file."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        results = {
            'experiment_name': self.experiment_name,
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'train_metrics': self.train_metrics,
            'test_metrics': self.test_metrics
        }
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.experiment_name}_{timestamp}.json"
        filepath = self.save_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nResults saved to: {filepath}")
        
        return filepath
    
    def print_summary(self):
        """Print experiment summary."""
        print("\n" + "="*60)
        print(f"Experiment: {self.experiment_name}")
        print("="*60)
        
        if self.train_metrics['epoch']:
            print("\nTraining Summary:")
            print(f"  Total epochs: {len(self.train_metrics['epoch'])}")
            print(f"  Final loss: {self.train_metrics['loss'][-1]:.4f}")
            print(f"  Final accuracy: {self.train_metrics['accuracy'][-1]:.4f}")
        
        if self.test_metrics:
            print("\nTest Results:")
            for key, value in self.test_metrics.items():
                print(f"  {key}: {value:.4f}")
        
        print("="*60 + "\n")


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str
):
    """Save model checkpoint.
    
    Args:
        model: PyTorch model.
        optimizer: Optimizer.
        epoch: Current epoch.
        loss: Current loss.
        filepath: Path to save checkpoint.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to: {filepath}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    filepath: str,
    device: str = 'cuda'
) -> int:
    """Load model checkpoint.
    
    Args:
        model: PyTorch model.
        optimizer: Optimizer (optional).
        filepath: Path to checkpoint.
        device: Device to load checkpoint to.
        
    Returns:
        Epoch number from checkpoint.
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', 0.0)
    
    print(f"Checkpoint loaded from: {filepath}")
    print(f"  Epoch: {epoch}, Loss: {loss:.4f}")
    
    return epoch


class AverageMeter:
    """Compute and store the average and current value.
    
    Useful for tracking metrics during training.
    """
    
    def __init__(self):
        """Initialize the meter."""
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """Update statistics.
        
        Args:
            val: Value to add.
            n: Number of samples.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0
