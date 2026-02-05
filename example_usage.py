"""Example usage script for HSI Few-Shot Classification.

This script demonstrates how to use the framework programmatically
with custom configurations and settings.
"""

import torch
from configs import ExperimentConfig, DataConfig, FewShotConfig, ModelConfig, TrainingConfig
from src.utils import set_random_seed


def example_1_basic_usage():
    """Example 1: Basic usage with default configuration."""
    print("\n" + "="*60)
    print("Example 1: Basic Usage")
    print("="*60)
    
    # Create default configuration
    config = ExperimentConfig()
    
    # Override specific settings
    config.data.data_path = "data/Indian_pines_corrected.mat"
    config.fewshot.n_way = 5
    config.fewshot.k_shot = 5
    config.training.num_epochs = 10
    
    print("Configuration created successfully!")
    print(f"  N-way: {config.fewshot.n_way}")
    print(f"  K-shot: {config.fewshot.k_shot}")
    print(f"  Epochs: {config.training.num_epochs}")


def example_2_custom_configuration():
    """Example 2: Creating custom configuration."""
    print("\n" + "="*60)
    print("Example 2: Custom Configuration")
    print("="*60)
    
    # Create custom data config
    data_config = DataConfig(
        data_path="data/Pavia_University.mat",
        hsi_key="pavia",
        gt_key="pavia_gt",
        n_components=40,  # More PCA components
        patch_size=11,    # Larger patches
        normalization="zscore",
        random_seed=123
    )
    
    # Create custom few-shot config
    fewshot_config = FewShotConfig(
        n_way=10,         # 10-way classification
        k_shot=3,         # 3-shot learning
        q_query=20,       # 20 query samples
        num_episodes_train=2000,
        num_episodes_test=200
    )
    
    # Create custom model config
    model_config = ModelConfig(
        conv_channels=(64, 128, 256),  # Larger network
        embedding_dim=512,
        dropout_rate=0.4
    )
    
    # Create custom training config
    training_config = TrainingConfig(
        learning_rate=0.0005,
        num_epochs=100,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Combine into experiment config
    config = ExperimentConfig(
        data=data_config,
        fewshot=fewshot_config,
        model=model_config,
        training=training_config,
        experiment_name="custom_experiment"
    )
    
    print("Custom configuration created!")
    print(f"  Experiment: {config.experiment_name}")
    print(f"  Patch size: {config.data.patch_size}")
    print(f"  PCA components: {config.data.n_components}")
    print(f"  Embedding dim: {config.model.embedding_dim}")


def example_3_data_loading():
    """Example 3: Data loading and preprocessing."""
    print("\n" + "="*60)
    print("Example 3: Data Loading (Standalone)")
    print("="*60)
    
    from src.data import HSIDataLoader, PatchExtractor
    
    # Note: This will fail without actual data, but shows the API
    try:
        loader = HSIDataLoader(
            data_path="data/Indian_pines_corrected.mat",
            hsi_key="indian_pines_corrected",
            gt_key="indian_pines_gt"
        )
        
        # Load data
        hsi_data, gt_data = loader.load_data()
        
        # Preprocess
        preprocessed = loader.preprocess(
            n_components=30,
            normalization="minmax"
        )
        
        # Extract patches
        patch_extractor = PatchExtractor(patch_size=9)
        patches, labels = patch_extractor.extract_patches(
            preprocessed,
            gt_data
        )
        
        print(f"Successfully loaded {len(patches)} patches!")
        
    except FileNotFoundError:
        print("⚠️  Data file not found (expected for demo)")
        print("This example shows the API for data loading.")


def example_4_model_creation():
    """Example 4: Creating and inspecting models."""
    print("\n" + "="*60)
    print("Example 4: Model Creation")
    print("="*60)
    
    from src.models import build_prototypical_network
    from src.utils import count_parameters
    
    # Create model
    model = build_prototypical_network(
        input_channels=1,
        spectral_depth=30,
        spatial_size=9,
        conv_channels=(32, 64, 128),
        embedding_dim=256,
        dropout_rate=0.3
    )
    
    # Count parameters
    num_params = count_parameters(model)
    
    print("Model created successfully!")
    print(f"  Total parameters: {num_params:,}")
    print(f"  Backbone type: 3D-CNN")
    print(f"  Meta-learner: Prototypical Network")
    
    # Test forward pass with dummy data
    dummy_input = torch.randn(5, 1, 30, 9, 9)  # 5 samples
    with torch.no_grad():
        features = model.extract_features(dummy_input)
    
    print(f"  Feature extraction test: {dummy_input.shape} → {features.shape}")


def example_5_reproducibility():
    """Example 5: Ensuring reproducibility."""
    print("\n" + "="*60)
    print("Example 5: Reproducibility")
    print("="*60)
    
    # Set seed
    set_random_seed(42)
    
    # Create identical random tensors
    tensor1 = torch.randn(3, 3)
    
    # Reset seed
    set_random_seed(42)
    tensor2 = torch.randn(3, 3)
    
    # Check if identical
    are_identical = torch.allclose(tensor1, tensor2)
    
    print(f"Reproducibility test: {'✓ PASSED' if are_identical else '✗ FAILED'}")
    print(f"  Tensor 1:\n{tensor1}")
    print(f"  Tensor 2:\n{tensor2}")
    print(f"  Identical: {are_identical}")


def example_6_metrics_calculation():
    """Example 6: Computing classification metrics."""
    print("\n" + "="*60)
    print("Example 6: Metrics Calculation")
    print("="*60)
    
    from src.utils import ClassificationMetrics
    import numpy as np
    
    # Create sample predictions (5-way classification, 100 samples)
    y_true = np.random.randint(0, 5, 100)
    y_pred = y_true.copy()
    # Introduce some errors
    error_indices = np.random.choice(100, 20, replace=False)
    y_pred[error_indices] = np.random.randint(0, 5, 20)
    
    # Compute metrics
    metrics = ClassificationMetrics.compute_all_metrics(
        y_true, y_pred, verbose=True
    )
    
    # Compute per-class accuracy
    per_class = ClassificationMetrics.compute_per_class_accuracy(
        y_true, y_pred
    )


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("HSI Few-Shot Classification - Example Usage")
    print("="*70)
    
    examples = [
        example_1_basic_usage,
        example_2_custom_configuration,
        example_3_data_loading,
        example_4_model_creation,
        example_5_reproducibility,
        example_6_metrics_calculation
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n⚠️  Example raised exception: {e}")
            print("(This is expected for data-dependent examples without real data)")
    
    print("\n" + "="*70)
    print("Examples completed!")
    print("="*70)
    print("\nKey Takeaways:")
    print("  1. Clean, modular configuration system")
    print("  2. Type-safe data loading and preprocessing")
    print("  3. Flexible model architecture")
    print("  4. Reproducible experiments with seed management")
    print("  5. Comprehensive metrics for remote sensing")
    print("\nReady to use with real HSI data!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
