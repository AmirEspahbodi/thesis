# Getting Started Guide

This guide will walk you through setting up and running your first few-shot HSI classification experiment.

## Table of Contents
1. [Installation](#installation)
2. [Dataset Preparation](#dataset-preparation)
3. [Quick Verification](#quick-verification)
4. [First Training Run](#first-training-run)
5. [Understanding the Output](#understanding-the-output)
6. [Next Steps](#next-steps)

## Installation

### Step 1: Clone or Download the Project

If you have this as a git repository:
```bash
git clone <repository-url>
cd hsi_fewshot_project
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch (deep learning framework)
- NumPy, SciPy (numerical computing)
- scikit-learn (PCA and metrics)
- Hydra (configuration management)
- tqdm (progress bars)

### Step 4: Verify Installation

Run the verification script:
```bash
python verify_setup.py
```

You should see something like:
```
✓ Python 3.x.x detected
✓ PyTorch installed
✓ CUDA available
✓ All checks passed!
```

## Dataset Preparation

### Supported Datasets

The framework supports 5 HSI datasets:
1. **Houston 2013** - 144 bands, 15 classes
2. **Houston 2018** - 48 bands, 20 classes  
3. **Indian Pines** - 200 bands, 16 classes
4. **Pavia University** - 103 bands, 9 classes
5. **Salinas** - 204 bands, 16 classes

### File Format

Each dataset needs two .mat files:
- **Image file**: Contains the hyperspectral cube (H × W × C)
- **Ground truth file**: Contains pixel labels (H × W)

### Directory Structure

Place your datasets in the following structure:
```
D:/work/thesis/dataset/
├── Houston13.mat
├── Houston13_7gt.mat
├── Houston18.mat
├── Houston18_7gt.mat
├── Indian_pines.mat
├── Indian_pines_gt.mat
├── PaviaU.mat
├── PaviaU_gt.mat
├── Salinas.mat
└── Salinas_gt.mat
```

**Note**: If your data is in a different location, update `paths.data_root` in `configs/config.yaml`.

### Verifying Dataset Keys

If you get "key not found" errors, check the actual keys in your .mat files:

```python
import scipy.io

# Load and inspect
data = scipy.io.loadmat('Houston13.mat')
print("Available keys:", data.keys())

# Update configs/dataset/houston13.yaml with correct keys:
# image_key: <the correct key>
# gt_key: <the correct key>
```

## Quick Verification

Before running with real data, test with synthetic data:

```bash
python test_synthetic.py
```

This will:
- Create synthetic HSI data
- Build the model
- Run 3 training epochs
- Verify the entire pipeline works

Expected output:
```
✓ Train dataset: 500 samples
✓ Val dataset: 200 samples
✓ Model created with X parameters
✓ Forward pass successful
Training completed successfully!
```

## First Training Run

### Example 1: In-Domain Learning on Houston 2013

```bash
python train.py dataset=houston13
```

**What happens:**
1. Loads Houston13 dataset
2. Applies PCA to reduce 144 bands → 30 bands
3. Splits data: 10% train, 10% val, 80% test
4. Creates episodic sampler (5-way 5-shot tasks)
5. Trains for 100 epochs with early stopping
6. Evaluates on test set
7. Saves results and best model

**Expected runtime:** ~10-20 minutes on GPU

### Example 2: Different Few-Shot Settings

Try 1-shot learning (harder):
```bash
python train.py dataset=houston13 few_shot.k_shot=1
```

Try 10-way (more classes):
```bash
python train.py dataset=houston13 few_shot.n_way=10
```

### Example 3: Cross-Domain Transfer

Train on Houston13, test on Houston18:
```bash
python train.py experiment=cross_domain
```

Train on Salinas, test on Indian Pines:
```bash
python train.py experiment=cross_domain_salinas_indian
```

## Understanding the Output

### During Training

You'll see progress bars like:
```
Epoch 1 [Train]: 100%|████████| 100/100 [00:30<00:00, loss: 1.234, acc: 0.456]
Epoch 1 [Val]: 100%|████████| 50/50 [00:10<00:00, loss: 1.123, acc: 0.478]

Train - Loss: 1.234, Acc: 0.456
Val   - Loss: 1.123, Acc: 0.478
✓ Saved best model (val_acc: 0.478)
```

**Metrics:**
- **Loss**: Lower is better (typical range: 0.5-2.0)
- **Accuracy**: Higher is better (0-1 scale)

### Final Evaluation

After training, you'll see:
```
==========================================
Classification Metrics
==========================================
Overall Accuracy (OA): 0.8245 (82.45%)
Average Accuracy (AA): 0.8123 (81.23%)
Kappa Coefficient:     0.7891

Per-Class Accuracy:
  Class  0: 0.8500 (85.00%)
  Class  1: 0.7800 (78.00%)
  ...
```

**Understanding Metrics:**
- **OA (Overall Accuracy)**: Total correct predictions
- **AA (Average Accuracy)**: Mean per-class accuracy (better for imbalanced data)
- **Kappa**: Agreement accounting for chance (> 0.8 is excellent)

### Output Files

```
outputs/
└── YYYY-MM-DD/
    └── HH-MM-SS/
        ├── .hydra/          # Hydra logs
        └── results.txt      # Final metrics

checkpoints/
└── best_model.pth           # Best model (highest val accuracy)
```

## Next Steps

### 1. Experiment with Hyperparameters

```bash
# Longer training
python train.py training.epochs=150

# Different learning rate
python train.py training.lr=0.0005

# Larger model
python train.py model.backbone.d_model=256

# Different patch size
python train.py data.patch_size=11
```

### 2. Try Different Datasets

```bash
# Indian Pines
python train.py dataset=indian_pines

# Salinas
python train.py dataset=salinas

# Pavia University
python train.py dataset=pavia_u
```

### 3. Custom Experiments

Create your own experiment config in `configs/experiment/`:

```yaml
# configs/experiment/my_experiment.yaml
defaults:
  - override /dataset: indian_pines

few_shot:
  n_way: 10
  k_shot: 1
  query_shot: 20

training:
  epochs: 150
  lr: 0.0005
```

Run it:
```bash
python train.py experiment=my_experiment
```

### 4. Add a New Dataset

1. Create config: `configs/dataset/my_data.yaml`
```yaml
file_name: MyData.mat
gt_name: MyData_gt.mat
image_key: my_image
gt_key: my_gt
n_bands: 100
target_bands: 30
ignored_labels: [0]
n_classes: 12
```

2. Run training:
```bash
python train.py dataset=my_data
```

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
python train.py training.batch_size=2

# Smaller model
python train.py model.backbone.d_model=64

# Fewer PCA bands
python train.py data.target_bands=20
```

### Poor Accuracy

- **Increase training data**: `data.train_ratio=0.2`
- **More epochs**: `training.epochs=150`
- **Lower learning rate**: `training.lr=0.0001`
- **Larger model**: `model.backbone.d_model=256`

### "Not enough samples" Error

```bash
# Reduce k-shot
python train.py few_shot.k_shot=3

# Increase train ratio
python train.py data.train_ratio=0.15
```

## Tips for Best Results

1. **Start small**: Begin with 5-way 5-shot
2. **Monitor validation**: If val acc >> train acc, you're overfitting
3. **PCA matters**: 30 bands is usually good, but you can experiment
4. **Patience**: Some datasets need 50+ epochs to converge
5. **Cross-domain is hard**: Lower accuracy is expected when transferring

## Support

If you encounter issues:
1. Run `python verify_setup.py` to check installation
2. Run `python test_synthetic.py` to verify pipeline
3. Check the dataset keys match your .mat files
4. Ensure sufficient VRAM (16GB recommended)

For further help, open an issue or contact the maintainer.

## Quick Reference

```bash
# Verify setup
python verify_setup.py

# Test with synthetic data
python test_synthetic.py

# Basic training
python train.py dataset=houston13

# 1-shot learning
python train.py dataset=houston13 few_shot.k_shot=1

# Cross-domain
python train.py experiment=cross_domain

# Custom parameters
python train.py \
    dataset=indian_pines \
    few_shot.n_way=10 \
    few_shot.k_shot=1 \
    training.epochs=150
```

Happy training! 🚀
