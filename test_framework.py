"""Validation script to test the framework structure.

This script validates that all modules are correctly structured,
imports work, and basic functionality is operational.
"""

import sys
from pathlib import Path


def test_imports():
    """Test that all modules can be imported."""
    print("\n" + "="*60)
    print("Test 1: Module Imports")
    print("="*60)
    
    tests_passed = []
    tests_failed = []
    
    # Test config imports
    try:
        from configs import (
            DataConfig, FewShotConfig, ModelConfig,
            TrainingConfig, ExperimentConfig
        )
        print("✓ Config imports successful")
        tests_passed.append("Config imports")
    except ImportError as e:
        print(f"✗ Config imports failed: {e}")
        tests_failed.append("Config imports")
    
    # Test data imports
    try:
        from src.data import (
            HSIDataLoader, PatchExtractor,
            HSIFewShotDataset, EpisodeSampler
        )
        print("✓ Data imports successful")
        tests_passed.append("Data imports")
    except ImportError as e:
        print(f"✗ Data imports failed: {e}")
        tests_failed.append("Data imports")
    
    # Test model imports
    try:
        from src.models import (
            HSI3DCNN, Conv3DBlock,
            PrototypicalNetwork, build_prototypical_network
        )
        print("✓ Model imports successful")
        tests_passed.append("Model imports")
    except ImportError as e:
        print(f"✗ Model imports failed: {e}")
        tests_failed.append("Model imports")
    
    # Test utils imports
    try:
        from src.utils import (
            ClassificationMetrics, set_random_seed,
            setup_logger, ExperimentTracker,
            count_parameters, AverageMeter
        )
        print("✓ Utils imports successful")
        tests_passed.append("Utils imports")
    except ImportError as e:
        print(f"✗ Utils imports failed: {e}")
        tests_failed.append("Utils imports")
    
    # Test training imports
    try:
        from src.training import FewShotTrainer
        print("✓ Training imports successful")
        tests_passed.append("Training imports")
    except ImportError as e:
        print(f"✗ Training imports failed: {e}")
        tests_failed.append("Training imports")
    
    return tests_passed, tests_failed


def test_configuration():
    """Test configuration system."""
    print("\n" + "="*60)
    print("Test 2: Configuration System")
    print("="*60)
    
    try:
        from configs import ExperimentConfig
        
        config = ExperimentConfig()
        
        # Test attribute access
        assert hasattr(config, 'data')
        assert hasattr(config, 'fewshot')
        assert hasattr(config, 'model')
        assert hasattr(config, 'training')
        
        # Test nested attributes
        assert hasattr(config.data, 'patch_size')
        assert hasattr(config.fewshot, 'n_way')
        assert hasattr(config.model, 'embedding_dim')
        assert hasattr(config.training, 'learning_rate')
        
        print("✓ Configuration system working correctly")
        print(f"  Data patch size: {config.data.patch_size}")
        print(f"  Few-shot N-way: {config.fewshot.n_way}")
        print(f"  Model embedding dim: {config.model.embedding_dim}")
        print(f"  Training LR: {config.training.learning_rate}")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_model_creation():
    """Test model creation and forward pass."""
    print("\n" + "="*60)
    print("Test 3: Model Creation")
    print("="*60)
    
    try:
        import torch
        from src.models import build_prototypical_network
        from src.utils import count_parameters
        
        # Create model
        model = build_prototypical_network(
            input_channels=1,
            spectral_depth=30,
            spatial_size=9,
            embedding_dim=128
        )
        
        # Test forward pass with dummy data
        dummy_support = torch.randn(10, 1, 30, 9, 9)  # 10 support samples
        dummy_query = torch.randn(15, 1, 30, 9, 9)    # 15 query samples
        support_labels = torch.randint(0, 5, (10,))    # 5-way
        
        with torch.no_grad():
            logits, prototypes = model(
                dummy_support, support_labels, dummy_query, n_way=5
            )
        
        # Validate shapes
        assert logits.shape == (15, 5), f"Expected (15, 5), got {logits.shape}"
        assert prototypes.shape == (5, 128), f"Expected (5, 128), got {prototypes.shape}"
        
        num_params = count_parameters(model)
        
        print("✓ Model creation successful")
        print(f"  Parameters: {num_params:,}")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Prototypes shape: {prototypes.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics():
    """Test metrics calculation."""
    print("\n" + "="*60)
    print("Test 4: Metrics Calculation")
    print("="*60)
    
    try:
        import numpy as np
        from src.utils import ClassificationMetrics
        
        # Create sample data
        y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        y_pred = np.array([0, 0, 1, 2, 2, 2, 3, 3, 4, 4])  # One error
        
        # Calculate metrics
        oa = ClassificationMetrics.overall_accuracy(y_true, y_pred)
        aa = ClassificationMetrics.average_accuracy(y_true, y_pred)
        kappa = ClassificationMetrics.kappa_coefficient(y_true, y_pred)
        
        # Validate
        assert 0 <= oa <= 1
        assert 0 <= aa <= 1
        assert -1 <= kappa <= 1
        
        print("✓ Metrics calculation successful")
        print(f"  Overall Accuracy: {oa:.4f}")
        print(f"  Average Accuracy: {aa:.4f}")
        print(f"  Kappa: {kappa:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Metrics test failed: {e}")
        return False


def test_episodic_sampling():
    """Test episodic sampler with dummy data."""
    print("\n" + "="*60)
    print("Test 5: Episodic Sampling")
    print("="*60)
    
    try:
        import numpy as np
        import torch
        from src.data import HSIFewShotDataset, EpisodeSampler
        
        # Create dummy data
        num_samples = 100
        patch_size = 9
        bands = 30
        
        patches = np.random.randn(num_samples, patch_size, patch_size, bands).astype(np.float32)
        labels = np.repeat(np.arange(10), 10)  # 10 classes, 10 samples each
        
        # Create dataset
        dataset = HSIFewShotDataset(patches, labels)
        
        # Create sampler
        sampler = EpisodeSampler(
            dataset=dataset,
            n_way=5,
            k_shot=3,
            q_query=5,
            num_episodes=10
        )
        
        # Sample an episode
        episode = sampler.sample_episode()
        
        # Validate episode structure
        assert 'support_data' in episode
        assert 'support_labels' in episode
        assert 'query_data' in episode
        assert 'query_labels' in episode
        
        # Validate shapes
        expected_support = 5 * 3  # n_way * k_shot
        expected_query = 5 * 5     # n_way * q_query
        
        assert episode['support_data'].shape[0] == expected_support
        assert episode['query_data'].shape[0] == expected_query
        
        print("✓ Episodic sampling successful")
        print(f"  Support set size: {episode['support_data'].shape[0]}")
        print(f"  Query set size: {episode['query_data'].shape[0]}")
        print(f"  Classes in episode: {episode['class_ids']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Episodic sampling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reproducibility():
    """Test reproducibility with seed setting."""
    print("\n" + "="*60)
    print("Test 6: Reproducibility")
    print("="*60)
    
    try:
        import torch
        import numpy as np
        from src.utils import set_random_seed
        
        # Set seed and generate random numbers
        set_random_seed(42)
        torch_tensor1 = torch.randn(3, 3)
        numpy_array1 = np.random.randn(3, 3)
        
        # Reset seed and regenerate
        set_random_seed(42)
        torch_tensor2 = torch.randn(3, 3)
        numpy_array2 = np.random.randn(3, 3)
        
        # Check reproducibility
        torch_match = torch.allclose(torch_tensor1, torch_tensor2)
        numpy_match = np.allclose(numpy_array1, numpy_array2)
        
        if torch_match and numpy_match:
            print("✓ Reproducibility test passed")
            print(f"  PyTorch reproducible: {torch_match}")
            print(f"  NumPy reproducible: {numpy_match}")
            return True
        else:
            print("✗ Reproducibility test failed")
            return False
            
    except Exception as e:
        print(f"✗ Reproducibility test failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("\n" + "="*70)
    print("HSI Few-Shot Classification Framework - Validation")
    print("="*70)
    
    # Run all tests
    all_tests = []
    
    # Test 1: Imports
    passed, failed = test_imports()
    all_tests.append(("Imports", len(failed) == 0))
    
    # Test 2: Configuration
    result = test_configuration()
    all_tests.append(("Configuration", result))
    
    # Test 3: Model
    result = test_model_creation()
    all_tests.append(("Model Creation", result))
    
    # Test 4: Metrics
    result = test_metrics()
    all_tests.append(("Metrics", result))
    
    # Test 5: Episodic Sampling
    result = test_episodic_sampling()
    all_tests.append(("Episodic Sampling", result))
    
    # Test 6: Reproducibility
    result = test_reproducibility()
    all_tests.append(("Reproducibility", result))
    
    # Summary
    print("\n" + "="*70)
    print("Validation Summary")
    print("="*70)
    
    total_tests = len(all_tests)
    passed_tests = sum(1 for _, passed in all_tests if passed)
    failed_tests = total_tests - passed_tests
    
    for test_name, passed in all_tests:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name:.<50} {status}")
    
    print("="*70)
    print(f"Results: {passed_tests}/{total_tests} tests passed")
    
    if failed_tests == 0:
        print("✓ All validation tests passed!")
        print("✓ Framework is ready to use!")
    else:
        print(f"✗ {failed_tests} test(s) failed")
        print("Please check the error messages above")
    
    print("="*70 + "\n")
    
    return failed_tests == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
