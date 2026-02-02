"""
Metrics Module for Hyperspectral Image Classification

Implements standard evaluation metrics:
- Overall Accuracy (OA)
- Average Accuracy (AA)
- Cohen's Kappa coefficient
"""

import numpy as np
from typing import Dict, Tuple
import torch
from sklearn.metrics import confusion_matrix, cohen_kappa_score


class AccuracyCalculator:
    """
    Calculate classification metrics for HSI

    Computes:
    - Overall Accuracy (OA): Percentage of correctly classified samples
    - Average Accuracy (AA): Mean of per-class accuracies
    - Kappa coefficient: Agreement measure accounting for chance
    """

    def __init__(self, n_classes: int):
        """
        Initialize calculator

        Args:
            n_classes: Number of classes (excluding background)
        """
        self.n_classes = n_classes
        self.reset()

    def reset(self):
        """Reset all stored predictions and labels"""
        self.all_predictions = []
        self.all_labels = []

    def update(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ):
        """
        Add batch of predictions and labels

        Args:
            predictions: (N,) tensor with predicted class indices
            labels: (N,) tensor with true class indices
        """
        # Convert to numpy and store
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        self.all_predictions.extend(predictions.tolist())
        self.all_labels.extend(labels.tolist())

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics

        Returns:
            Dictionary containing:
            - 'OA': Overall Accuracy
            - 'AA': Average Accuracy
            - 'Kappa': Cohen's Kappa coefficient
            - 'per_class_accuracy': Per-class accuracies
        """
        predictions = np.array(self.all_predictions)
        labels = np.array(self.all_labels)

        # Overall Accuracy
        oa = np.mean(predictions == labels)

        # Confusion matrix
        cm = confusion_matrix(labels, predictions, labels=range(self.n_classes))

        # Per-class accuracy
        per_class_acc = []
        for i in range(self.n_classes):
            if cm[i, :].sum() > 0:
                class_acc = cm[i, i] / cm[i, :].sum()
                per_class_acc.append(class_acc)
            else:
                per_class_acc.append(0.0)

        # Average Accuracy
        aa = np.mean(per_class_acc)

        # Kappa coefficient
        kappa = cohen_kappa_score(labels, predictions)

        return {
            "OA": oa,
            "AA": aa,
            "Kappa": kappa,
            "per_class_accuracy": per_class_acc,
            "confusion_matrix": cm,
        }

    def get_summary(self) -> str:
        """
        Get formatted summary of metrics

        Returns:
            Formatted string with all metrics
        """
        metrics = self.compute()

        summary = [
            "=" * 50,
            "Classification Metrics",
            "=" * 50,
            f"Overall Accuracy (OA): {metrics['OA']:.4f} ({metrics['OA'] * 100:.2f}%)",
            f"Average Accuracy (AA): {metrics['AA']:.4f} ({metrics['AA'] * 100:.2f}%)",
            f"Kappa Coefficient:     {metrics['Kappa']:.4f}",
            "",
            "Per-Class Accuracy:",
        ]

        for i, acc in enumerate(metrics["per_class_accuracy"]):
            summary.append(f"  Class {i:2d}: {acc:.4f} ({acc * 100:.2f}%)")

        summary.append("=" * 50)

        return "\n".join(summary)


def compute_episode_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Compute accuracy for a single episode

    Args:
        predictions: (N,) tensor with predicted class indices
        labels: (N,) tensor with true class indices

    Returns:
        Accuracy as float in [0, 1]
    """
    correct = (predictions == labels).float().sum()
    total = len(labels)
    accuracy = correct / total

    return accuracy.item()


def compute_class_balanced_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
) -> Tuple[float, np.ndarray]:
    """
    Compute class-balanced accuracy (same as AA)

    Args:
        predictions: (N,) tensor with predicted class indices
        labels: (N,) tensor with true class indices
        n_classes: Number of classes

    Returns:
        balanced_accuracy: Mean of per-class accuracies
        per_class_acc: Array of per-class accuracies
    """
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()

    per_class_acc = []
    for i in range(n_classes):
        mask = labels == i
        if mask.sum() > 0:
            class_correct = ((predictions == labels) & mask).sum()
            class_total = mask.sum()
            class_acc = class_correct / class_total
            per_class_acc.append(class_acc)
        else:
            per_class_acc.append(0.0)

    per_class_acc = np.array(per_class_acc)
    balanced_accuracy = per_class_acc.mean()

    return balanced_accuracy, per_class_acc


class RunningAverage:
    """
    Compute running average of a metric

    Useful for tracking metrics during training
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the average"""
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        """
        Update the average with new value(s)

        Args:
            value: Value to add
            n: Number of samples (weight)
        """
        self.total += value * n
        self.count += n

    def get(self) -> float:
        """Get the current average"""
        if self.count == 0:
            return 0.0
        return self.total / self.count


if __name__ == "__main__":
    # Test the metrics calculator
    n_classes = 5
    n_samples = 100

    # Generate random predictions and labels
    predictions = torch.randint(0, n_classes, (n_samples,))
    labels = torch.randint(0, n_classes, (n_samples,))

    # Create calculator
    calc = AccuracyCalculator(n_classes)

    # Update with predictions
    calc.update(predictions, labels)

    # Compute metrics
    metrics = calc.compute()

    print(calc.get_summary())

    # Test episode accuracy
    episode_acc = compute_episode_accuracy(predictions, labels)
    print(f"\nEpisode Accuracy: {episode_acc:.4f}")

    # Test balanced accuracy
    balanced_acc, per_class = compute_class_balanced_accuracy(
        predictions, labels, n_classes
    )
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
