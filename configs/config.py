"""Configuration classes for HSI Few-Shot Learning.

This module contains dataclasses for organizing all hyperparameters,
paths, and training configurations.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class Houston13Dataset:
    # Houston 2013 Dataset Configuration
    file_name = "Houston13.mat"
    gt_name = "Houston13_7gt.mat"
    image_key = "ori_data"
    gt_key = "map"
    n_bands = 144
    target_bands = 30
    ignored_labels = [0]
    n_classes = 15


@dataclass
class Houston18Dataset:
    file_name = "Houston18.mat"
    gt_name = "Houston18_7gt.mat"
    image_key = "ori_data"
    gt_key = "map"
    n_bands = 48
    target_bands = 30
    ignored_labels = [0]
    n_classes = 20


@dataclass
class IndianPines:
    file_name = "Indian_pines.mat"
    gt_name = "Indian_pines_gt.mat"
    image_key = "indian_pines"
    gt_key = "indian_pines_gt"
    n_bands = 200
    target_bands = 30
    ignored_labels = [0]
    n_classes = 16


@dataclass
class PaviaUniversity:
    file_name = "PaviaU.mat"
    gt_name = "PaviaU_gt.mat"
    image_key = "paviaU"
    gt_key = "paviaU_gt"
    n_bands = 103
    target_bands = 30
    ignored_labels = [0]
    n_classes = 9


@dataclass
class Salinas:
    file_name = "Salinas.mat"
    gt_name = "Salinas_gt.mat"
    image_key = "salinas"
    gt_key = "salinas_gt"
    n_bands = 204
    target_bands = 30
    ignored_labels = [0]
    n_classes = 16


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing.

    Attributes:
        data_path: Path to the .mat file containing HSI data.
        hsi_key: Key name for HSI cube in .mat file.
        gt_key: Key name for ground truth labels in .mat file.
        n_components: Number of PCA components for spectral reduction.
        patch_size: Spatial size of the extracted patches (e.g., 9 for 9x9).
        normalization: Type of normalization ('minmax' or 'zscore').
        train_ratio: Ratio of samples used for training (rest for testing).
        random_seed: Random seed for reproducibility.
    """

    root_data_dir: str = "./dataset/"
    dataset_conf = PaviaUniversity
    n_components: int = 30
    patch_size: int = 9
    normalization: str = "minmax"
    train_ratio: float = 0.8
    random_seed: int = 42


@dataclass
class FewShotConfig:
    """Configuration for few-shot learning setup.

    Attributes:
        n_way: Number of classes per episode.
        k_shot: Number of support samples per class.
        q_query: Number of query samples per class.
        num_episodes_train: Number of training episodes.
        num_episodes_test: Number of testing episodes.
        source_classes: List of class IDs for training (source domain).
        target_classes: List of class IDs for testing (target domain).
    """

    n_way: int = 5
    k_shot: int = 5
    q_query: int = 15
    num_episodes_train: int = 1000
    num_episodes_test: int = 100
    source_classes: Optional[list] = None
    target_classes: Optional[list] = None


@dataclass
class ModelConfig:
    """Configuration for model architecture.

    Attributes:
        input_channels: Number of input channels (1 for single HSI cube).
        spectral_depth: Depth of spectral dimension after PCA.
        conv_channels: List of channel sizes for Conv3D layers.
        kernel_sizes: List of kernel sizes for each Conv3D layer.
        pool_sizes: List of pooling sizes for each layer.
        embedding_dim: Dimension of the final embedding vector.
        dropout_rate: Dropout rate for regularization.
    """

    input_channels: int = 1
    spectral_depth: int = 30
    conv_channels: Tuple[int, ...] = (32, 64, 128)
    kernel_sizes: Tuple[Tuple[int, int, int], ...] = ((3, 3, 3), (3, 3, 3), (3, 3, 3))
    pool_sizes: Tuple[Tuple[int, int, int], ...] = ((2, 2, 2), (2, 2, 2), (2, 2, 2))
    embedding_dim: int = 256
    dropout_rate: float = 0.3


@dataclass
class TrainingConfig:
    """Configuration for training process.

    Attributes:
        learning_rate: Learning rate for optimizer.
        weight_decay: L2 regularization parameter.
        num_epochs: Number of training epochs.
        device: Device to use for training ('cuda' or 'cpu').
        checkpoint_dir: Directory to save model checkpoints.
        log_interval: Interval (in episodes) for logging.
        save_best_only: Whether to save only the best model.
    """

    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    num_epochs: int = 50
    device: str = "cuda"
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 50
    save_best_only: bool = True


@dataclass
class ExperimentConfig:
    """Main configuration aggregating all sub-configs.

    Attributes:
        data: Data configuration.
        fewshot: Few-shot learning configuration.
        model: Model architecture configuration.
        training: Training process configuration.
        experiment_name: Name of the experiment for logging.
    """

    data: DataConfig = field(default_factory=DataConfig)
    fewshot: FewShotConfig = field(default_factory=FewShotConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment_name: str = "hsi_fewshot_exp1"

    def __post_init__(self):
        """Post-initialization to create checkpoint directory."""
        Path(self.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)
