"""Metrics calculation for remote sensing classification.

This module implements standard remote sensing classification metrics:
Overall Accuracy (OA), Average Accuracy (AA), and Kappa Coefficient.
"""

import numpy as np
import torch
from typing import Tuple, Dict
from sklearn.metrics import confusion_matrix, cohen_kappa_score


class ClassificationMetrics:
    """Calculate classification metrics for remote sensing.
    
    This class computes OA, AA, and Kappa coefficient from predictions
    and ground truth labels.
    """
    
    @staticmethod
    def overall_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Overall Accuracy (OA).
        
        OA is the ratio of correctly classified samples to total samples.
        
        Args:
            y_true: Ground truth labels of shape (N,).
            y_pred: Predicted labels of shape (N,).
            
        Returns:
            Overall accuracy as a float between 0 and 1.
        """
        correct = np.sum(y_true == y_pred)
        total = len(y_true)
        return float(correct / total)
    
    @staticmethod
    def average_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Average Accuracy (AA).
        
        AA is the mean of per-class accuracies.
        
        Args:
            y_true: Ground truth labels of shape (N,).
            y_pred: Predicted labels of shape (N,).
            
        Returns:
            Average accuracy as a float between 0 and 1.
        """
        # Get unique classes
        classes = np.unique(y_true)
        
        class_accuracies = []
        
        for class_id in classes:
            # Get indices for this class
            class_mask = y_true == class_id
            
            # Calculate accuracy for this class
            class_correct = np.sum(y_pred[class_mask] == class_id)
            class_total = np.sum(class_mask)
            
            if class_total > 0:
                class_acc = class_correct / class_total
                class_accuracies.append(class_acc)
        
        # Return mean of class accuracies
        return float(np.mean(class_accuracies))
    
    @staticmethod
    def kappa_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Kappa Coefficient (κ).
        
        Kappa measures inter-rater agreement, accounting for chance agreement.
        
        Args:
            y_true: Ground truth labels of shape (N,).
            y_pred: Predicted labels of shape (N,).
            
        Returns:
            Kappa coefficient as a float between -1 and 1.
        """
        return float(cohen_kappa_score(y_true, y_pred))
    
    @staticmethod
    def confusion_matrix_stats(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Calculate confusion matrix and per-class statistics.
        
        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            
        Returns:
            Tuple of (confusion_matrix, stats_dict) where stats_dict contains:
                - precision: Per-class precision
                - recall: Per-class recall
                - f1_score: Per-class F1 score
        """
        # Get confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate per-class metrics
        num_classes = cm.shape[0]
        precision = np.zeros(num_classes)
        recall = np.zeros(num_classes)
        f1_score = np.zeros(num_classes)
        
        for i in range(num_classes):
            # True positives
            tp = cm[i, i]
            
            # False positives
            fp = np.sum(cm[:, i]) - tp
            
            # False negatives
            fn = np.sum(cm[i, :]) - tp
            
            # Precision
            if (tp + fp) > 0:
                precision[i] = tp / (tp + fp)
            
            # Recall
            if (tp + fn) > 0:
                recall[i] = tp / (tp + fn)
            
            # F1 Score
            if (precision[i] + recall[i]) > 0:
                f1_score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
        
        stats = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
        
        return cm, stats
    
    @classmethod
    def compute_all_metrics(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, float]:
        """Compute all classification metrics.
        
        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            verbose: Whether to print metrics.
            
        Returns:
            Dictionary containing all metrics.
        """
        # Convert torch tensors to numpy if needed
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.cpu().numpy()
        
        # Calculate metrics
        oa = cls.overall_accuracy(y_true, y_pred)
        aa = cls.average_accuracy(y_true, y_pred)
        kappa = cls.kappa_coefficient(y_true, y_pred)
        
        metrics = {
            'overall_accuracy': oa,
            'average_accuracy': aa,
            'kappa_coefficient': kappa
        }
        
        if verbose:
            print("\n" + "="*50)
            print("Classification Metrics:")
            print("="*50)
            print(f"Overall Accuracy (OA): {oa:.4f} ({oa*100:.2f}%)")
            print(f"Average Accuracy (AA): {aa:.4f} ({aa*100:.2f}%)")
            print(f"Kappa Coefficient (κ): {kappa:.4f}")
            print("="*50 + "\n")
        
        return metrics
    
    @classmethod
    def compute_per_class_accuracy(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Dict[int, str] = None
    ) -> Dict[int, float]:
        """Compute per-class accuracy.
        
        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            class_names: Optional mapping from class IDs to names.
            
        Returns:
            Dictionary mapping class IDs to accuracies.
        """
        classes = np.unique(y_true)
        per_class_acc = {}
        
        print("\nPer-Class Accuracy:")
        print("-" * 50)
        
        for class_id in classes:
            class_mask = y_true == class_id
            class_correct = np.sum(y_pred[class_mask] == class_id)
            class_total = np.sum(class_mask)
            
            if class_total > 0:
                acc = class_correct / class_total
                per_class_acc[int(class_id)] = acc
                
                class_name = class_names.get(class_id, f"Class {class_id}") if class_names else f"Class {class_id}"
                print(f"{class_name}: {acc:.4f} ({acc*100:.2f}%) - {class_correct}/{class_total} samples")
        
        print("-" * 50)
        
        return per_class_acc
