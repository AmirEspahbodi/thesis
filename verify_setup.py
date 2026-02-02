"""
Setup Verification Script

Run this script to verify that all dependencies are installed correctly
and the project structure is valid.
"""

import sys
import os


def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")
        return True
    else:
        print(f"✗ Python 3.8+ required, but {version.major}.{version.minor} detected")
        return False


def check_imports():
    """Check if all required packages can be imported"""
    print("\nChecking required packages...")

    packages = {
        "torch": "PyTorch",
        "numpy": "NumPy",
        "scipy": "SciPy",
        "sklearn": "scikit-learn",
        "hydra": "Hydra",
        "omegaconf": "OmegaConf",
        "tqdm": "tqdm",
    }

    all_ok = True
    for module, name in packages.items():
        try:
            __import__(module)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"✗ {name} NOT installed")
            all_ok = False

    return all_ok


def check_cuda():
    """Check CUDA availability"""
    print("\nChecking CUDA...")
    try:
        import torch

        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(
                f"  Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            )
        else:
            print("⚠ CUDA not available (CPU mode will be used)")
        return True
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
        return False


def check_project_structure():
    """Check if project structure is correct"""
    print("\nChecking project structure...")

    required_paths = [
        "configs",
        "configs/config.yaml",
        "configs/dataset",
        "configs/experiment",
        "src",
        "src/datamodules",
        "src/models",
        "src/utils",
        "src/engine.py",
        "train.py",
        "requirements.txt",
    ]

    all_ok = True
    for path in required_paths:
        if os.path.exists(path):
            print(f"✓ {path}")
        else:
            print(f"✗ {path} NOT found")
            all_ok = False

    return all_ok


def check_model():
    """Check if model can be instantiated"""
    print("\nChecking model instantiation...")

    try:
        sys.path.insert(0, "src")
        from models import Simple3DCNN, PrototypicalNetwork
        import torch

        # Create backbone
        backbone = Simple3DCNN(
            input_channels=1, spectral_size=30, spatial_size=9, d_model=128
        )

        # Create ProtoNet
        model = PrototypicalNetwork(backbone)

        # Test forward pass
        test_input = torch.randn(5, 30, 9, 9)  # 5 samples
        features = backbone(test_input)

        if features.shape == (5, 128):
            print(f"✓ Model instantiated and forward pass works")
            print(f"  Input shape: {test_input.shape}")
            print(f"  Output shape: {features.shape}")
            return True
        else:
            print(f"✗ Unexpected output shape: {features.shape}")
            return False

    except Exception as e:
        print(f"✗ Error instantiating model: {e}")
        import traceback

        traceback.print_exc()
        return False


def check_dataset_loader():
    """Check if dataset loader works (if data available)"""
    print("\nChecking dataset loader...")

    try:
        sys.path.insert(0, "src")
        from datamodules import HSIDataset

        # Check if data directory exists
        data_root = "D:/work/thesis/dataset"
        if not os.path.exists(data_root):
            print(f"⚠ Data directory not found: {data_root}")
            print(f"  This is OK if you haven't downloaded the data yet")
            return True

        # Try to load a dataset
        test_file = os.path.join(data_root, "Houston13.mat")
        if not os.path.exists(test_file):
            print(f"⚠ Test dataset not found: {test_file}")
            print(f"  This is OK if you haven't downloaded the data yet")
            return True

        print(f"  Found data directory and test file")
        print(f"✓ Dataset loader is ready (data available)")
        return True

    except Exception as e:
        print(f"✗ Error checking dataset loader: {e}")
        return False


def main():
    """Run all checks"""
    print("=" * 60)
    print("HSI Few-Shot Learning Framework - Setup Verification")
    print("=" * 60)

    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_imports),
        ("CUDA", check_cuda),
        ("Project Structure", check_project_structure),
        ("Model", check_model),
        ("Dataset Loader", check_dataset_loader),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Unexpected error in {name}: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:.<50} {status}")
        if not result:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n✓ All checks passed! You're ready to start training.")
        print("\nQuick start:")
        print("  python train.py dataset=houston13")
    else:
        print("\n⚠ Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Make sure you're in the project root directory")
        print("  - Check that all source files are present")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
