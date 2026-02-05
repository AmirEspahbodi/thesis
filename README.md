# Hyperspectral Image Few-Shot Classification Framework

A production-grade, modular Python framework for **Few-Shot Learning (FSL)** on **Hyperspectral Images (HSI)** using **Prototypical Networks** and **3D-CNN** backbones.

## 🎯 Overview

This framework implements a complete pipeline for few-shot classification of hyperspectral remote sensing data, following Clean Architecture principles and industry best practices.

### Key Features

- ✅ **Modular Architecture**: Clean separation of concerns across data, models, training, and utilities
- ✅ **3D-CNN Backbone**: Optimized for spatial-spectral feature extraction from HSI cubes
- ✅ **Prototypical Networks**: Meta-learning approach for few-shot classification
- ✅ **Episodic Sampling**: Proper N-way K-shot task generation with no data leakage
- ✅ **Remote Sensing Metrics**: OA, AA, and Kappa coefficient
- ✅ **Type Hints & Docstrings**: Comprehensive documentation throughout
- ✅ **Reproducibility**: Seed management for consistent results
- ✅ **Production Ready**: Error handling, logging, checkpointing

## 📁 Project Structure

```
hsi_fewshot/
├── configs/
│   ├── __init__.py
│   └── config.py              # Configuration dataclasses
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py          # HSI data loading & preprocessing
│   │   └── dataset.py         # PyTorch datasets & episodic sampler
│   ├── models/
│   │   ├── __init__.py
│   │   ├── backbone.py        # 3D-CNN feature extractor
│   │   └── protonet.py        # Prototypical Network
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── metrics.py         # Classification metrics (OA, AA, Kappa)
│   │   └── helpers.py         # Logging, checkpointing, utilities
│   └── training/
│       ├── __init__.py
│       └── trainer.py         # Training engine
├── main.py                    # Entry point
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone or download the project
cd hsi_fewshot

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

Place your hyperspectral data (`.mat` format) in a `data/` directory:

```
data/
├── Indian_pines_corrected.mat
├── Pavia_University.mat
└── Houston.mat
```

Expected `.mat` file structure:
- HSI cube: `(Height, Width, Bands)` - e.g., `(145, 145, 200)`
- Ground truth: `(Height, Width)` - integer labels

### Basic Usage

```bash
# Run with default configuration
python main.py

# Custom parameters
python main.py --data_path data/Pavia_University.mat \
               --n_way 5 \
               --k_shot 5 \
               --num_epochs 50 \
               --device cuda
```

### Command Line Arguments

```
--data_path     Path to HSI .mat file (default: data/Indian_pines_corrected.mat)
--n_way         Number of classes per episode (default: 5)
--k_shot        Support samples per class (default: 5)
--num_epochs    Number of training epochs (default: 50)
--seed          Random seed (default: 42)
--device        Device: 'cuda' or 'cpu' (default: auto-detect)
```

## 🧠 Technical Details

### Data Pipeline

1. **Loading**: Reads HSI cube and ground truth from `.mat` files
2. **PCA**: Reduces spectral bands (e.g., 200 → 30) while preserving variance
3. **Normalization**: Min-Max or Z-score normalization
4. **Patching**: Extracts 3D spatial-spectral cubes (e.g., 9×9×30) with boundary padding

### Model Architecture

#### 3D-CNN Backbone
- **Input**: `(Batch, 1, Spectral_Depth, Height, Width)`
- **Architecture**: Stacked `Conv3D → BatchNorm → ReLU → MaxPool3D`
- **Output**: Fixed-dimensional embeddings (e.g., 256-D)

#### Prototypical Network
- **Support Set**: Computes class prototypes (mean embeddings)
- **Query Set**: Classifies based on Euclidean distance to prototypes
- **Loss**: Cross-entropy on negative distances

### Few-Shot Setup

**Domain Disjoint Split**:
- **Source Domain**: Classes used for training (e.g., first 60%)
- **Target Domain**: Classes used for testing (e.g., remaining 40%)
- No overlap between source and target classes

**Episode Structure**:
- **N-way**: Number of classes (e.g., 5)
- **K-shot**: Support samples per class (e.g., 5)
- **Q-query**: Query samples per class (e.g., 15)

### Evaluation Metrics

1. **Overall Accuracy (OA)**: `correct_samples / total_samples`
2. **Average Accuracy (AA)**: Mean of per-class accuracies
3. **Kappa Coefficient (κ)**: Agreement accounting for chance

## 📊 Configuration

Modify `configs/config.py` or create custom configurations:

```python
from configs import ExperimentConfig

config = ExperimentConfig()

# Data settings
config.data.n_components = 30          # PCA components
config.data.patch_size = 9             # 9x9 spatial patches
config.data.normalization = "minmax"   # or "zscore"

# Few-shot settings
config.fewshot.n_way = 5               # 5-way classification
config.fewshot.k_shot = 5              # 5 support samples
config.fewshot.q_query = 15            # 15 query samples

# Model settings
config.model.conv_channels = (32, 64, 128)
config.model.embedding_dim = 256

# Training settings
config.training.learning_rate = 0.001
config.training.num_epochs = 50
```

## 🔬 Example Results

### Indian Pines Dataset (Example)
```
Classification Metrics:
==================================================
Overall Accuracy (OA): 0.8234 (82.34%)
Average Accuracy (AA): 0.7956 (79.56%)
Kappa Coefficient (κ): 0.7845
==================================================
```

## 🛠️ Advanced Usage

### Custom Datasets

To use with your own HSI data:

1. Ensure `.mat` format with keys for HSI cube and labels
2. Update `config.data.hsi_key` and `config.data.gt_key`
3. Adjust `config.data.n_components` based on your spectral bands

### Custom Models

Modify the backbone in `src/models/backbone.py`:

```python
# Example: Add more convolutional layers
config.model.conv_channels = (32, 64, 128, 256)
config.model.kernel_sizes = ((3,3,3), (3,3,3), (3,3,3), (3,3,3))
```

### Checkpointing

Models are automatically saved to `checkpoints/`:
- `best_model.pt`: Best performing model
- `model_epoch_N.pt`: Epoch-specific checkpoints

Load a checkpoint:

```python
from src.utils import load_checkpoint

epoch = load_checkpoint(model, optimizer, 'checkpoints/best_model.pt')
```

## 📝 Code Quality

- **Type Hints**: All functions use Python typing
- **Docstrings**: Google-style documentation
- **Error Handling**: Try-except blocks for robustness
- **Modularity**: Single Responsibility Principle
- **Reproducibility**: Seed management across all libraries

## 🧪 Testing the Framework

```bash
# Test without real data (structure validation)
python main.py

# Test with sample data
python main.py --data_path data/sample.mat --num_epochs 5
```

## 📚 References

### Key Papers

1. **Prototypical Networks**: Snell et al., "Prototypical Networks for Few-shot Learning", NeurIPS 2017
2. **3D-CNN for HSI**: Li et al., "Deep Learning for Hyperspectral Image Classification: An Overview", IEEE TGRS 2019

### Datasets

- **Indian Pines**: [Purdue HSI Dataset](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)
- **Pavia University**: IEEE GRSS Data Fusion Contest
- **Houston**: IEEE GRSS Data Fusion Contest 2013

## 🤝 Contributing

This framework is designed to be extended. Key extension points:

1. **New Backbones**: Add to `src/models/backbone.py`
2. **New Meta-Learners**: Add to `src/models/`
3. **New Metrics**: Add to `src/utils/metrics.py`
4. **Data Augmentation**: Extend `src/data/loader.py`

## 📄 License

This is a research framework. Please cite appropriately if used in publications.

## 🐛 Troubleshooting

**CUDA Out of Memory**:
```bash
# Reduce batch size (queries per episode)
config.fewshot.q_query = 10  # instead of 15
```

**File Not Found**:
```bash
# Verify data path
ls data/Indian_pines_corrected.mat
```

**Import Errors**:
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

## 📧 Support

For issues or questions about the framework structure, check:
1. Type hints and docstrings in the code
2. Configuration examples in `configs/config.py`
3. Main pipeline in `main.py`

---

**Built with Clean Architecture principles for production-grade ML research.**
