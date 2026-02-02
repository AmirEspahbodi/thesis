# Few-Shot Hyperspectral Image Classification Framework

A production-ready PyTorch framework for Few-Shot Hyperspectral Image (HSI) Classification using Prototypical Networks with 3D-CNN backbone.

## Features

- ✅ **Clean Architecture**: Modular design following SOLID principles
- ✅ **Prototypical Networks**: State-of-the-art few-shot learning algorithm
- ✅ **3D-CNN Backbone**: Spatial-spectral feature extraction
- ✅ **PCA Preprocessing**: Automatic spectral dimension reduction
- ✅ **Dual Strategies**: In-domain and cross-domain learning
- ✅ **Hydra Configuration**: Flexible experiment management
- ✅ **Comprehensive Metrics**: OA, AA, and Kappa coefficient
- ✅ **16GB VRAM Optimized**: Efficient memory usage

## Project Structure

```
project_root/
├── configs/
│   ├── config.yaml              # Main configuration
│   ├── dataset/                 # Dataset-specific configs
│   │   ├── houston13.yaml
│   │   ├── houston18.yaml
│   │   ├── indian_pines.yaml
│   │   ├── pavia_u.yaml
│   │   └── salinas.yaml
│   └── experiment/              # Experiment configs
│       ├── in_domain.yaml
│       └── cross_domain.yaml
├── src/
│   ├── datamodules/
│   │   ├── hsi_dataset.py       # Dataset loader with PCA
│   │   └── samplers.py          # Few-shot episodic sampler
│   ├── models/
│   │   ├── backbone.py          # 3D-CNN feature extractor
│   │   └── protonet.py          # Prototypical Network
│   ├── utils/
│   │   └── metrics.py           # Evaluation metrics
│   └── engine.py                # Training and evaluation
├── train.py                     # Main entry point
└── requirements.txt
```

## Installation

### 1. Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ VRAM

### 2. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Setup

Place your HSI datasets in the following structure:

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

## Usage

### Basic Training

#### In-Domain Learning (Train/Test on same dataset)

```bash
# Train on Houston 2013
python train.py dataset=houston13

# Train on Indian Pines
python train.py dataset=indian_pines

# Train on Salinas
python train.py dataset=salinas
```

#### Cross-Domain Learning (Transfer learning)

```bash
# Houston13 -> Houston18
python train.py experiment=cross_domain

# Custom cross-domain (modify configs/experiment/cross_domain.yaml)
```

### Advanced Configuration

#### Change Few-Shot Settings

```bash
# 5-way 1-shot
python train.py few_shot.k_shot=1

# 10-way 5-shot
python train.py few_shot.n_way=10 few_shot.k_shot=5

# More query samples
python train.py few_shot.query_shot=20
```

#### Modify Training Parameters

```bash
# Longer training
python train.py training.epochs=150

# Adjust learning rate
python train.py training.lr=0.0005

# Change batch size
python train.py training.batch_size=8
```

#### Model Architecture

```bash
# Larger feature dimension
python train.py model.backbone.d_model=256

# Different patch size
python train.py data.patch_size=11

# More/fewer PCA components
python train.py data.target_bands=50
```

### Multiple Configurations

```bash
# Combine multiple overrides
python train.py \
    dataset=indian_pines \
    few_shot.n_way=5 \
    few_shot.k_shot=1 \
    training.epochs=100 \
    training.lr=0.001
```

## Configuration Details

### Main Config (`configs/config.yaml`)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `few_shot.n_way` | Classes per episode | 5 |
| `few_shot.k_shot` | Support samples per class | 5 |
| `few_shot.query_shot` | Query samples per class | 15 |
| `data.patch_size` | Spatial patch size | 9 |
| `data.target_bands` | Spectral bands after PCA | 30 |
| `training.epochs` | Training epochs | 100 |
| `training.lr` | Learning rate | 0.001 |

### Dataset Config Example (`configs/dataset/houston13.yaml`)

```yaml
file_name: Houston13.mat
gt_name: Houston13_7gt.mat
image_key: Houston13
gt_key: Houston13_7gt
n_bands: 144
target_bands: 30
ignored_labels: [0]
n_classes: 15
```

## Output Structure

After training, the following files are generated:

```
outputs/
└── YYYY-MM-DD/
    └── HH-MM-SS/
        └── results.txt          # Final test metrics

checkpoints/
└── best_model.pth               # Best model checkpoint
```

## Model Architecture

### 3D-CNN Backbone

```
Input: (B, 1, 30, 9, 9)
    ↓
Conv3D(1→8) + BN + ReLU
    ↓ (B, 8, 12, 9, 9)
Conv3D(8→16) + BN + ReLU
    ↓ (B, 16, 8, 9, 9)
Conv3D(16→32) + BN + ReLU
    ↓ (B, 32, 5, 9, 9)
GlobalAvgPool3D
    ↓ (B, 32, 1, 1, 1)
FC(32→128)
    ↓
Output: (B, 128)
```

### Prototypical Network

1. **Feature Extraction**: Extract features from support and query sets
2. **Prototype Computation**: Compute class prototypes (mean of support features)
3. **Distance Calculation**: Calculate Euclidean distance between queries and prototypes
4. **Classification**: Assign query to nearest prototype

## Evaluation Metrics

The framework computes the following metrics:

- **Overall Accuracy (OA)**: Percentage of correctly classified samples
- **Average Accuracy (AA)**: Mean of per-class accuracies (handles class imbalance)
- **Kappa Coefficient**: Agreement measure accounting for chance
- **Per-Class Accuracy**: Individual accuracy for each class

## Memory Optimization

The framework is optimized for 16GB VRAM:

- Efficient 3D-CNN architecture (lightweight layers)
- Batch size of 4 episodes
- PCA reduces spectral dimensions (200+ → 30 bands)
- Small patch size (9×9)
- No gradient accumulation needed

If you encounter OOM errors:
```bash
# Reduce batch size
python train.py training.batch_size=2

# Smaller feature dimension
python train.py model.backbone.d_model=64

# Fewer PCA components
python train.py data.target_bands=20
```

## Extending the Framework

### Adding a New Dataset

1. Create config file: `configs/dataset/my_dataset.yaml`
```yaml
file_name: MyDataset.mat
gt_name: MyDataset_gt.mat
image_key: my_data
gt_key: my_gt
n_bands: 100
target_bands: 30
ignored_labels: [0]
n_classes: 10
```

2. Run training:
```bash
python train.py dataset=my_dataset
```

### Adding Attention Mechanism

The framework includes a placeholder for attention-based enhancements:

```python
from models import PrototypicalNetworkWithAttention

model = PrototypicalNetworkWithAttention(
    backbone=backbone,
    d_model=128,
    n_heads=4
)
```

Implement custom attention in `src/models/protonet.py`.

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size or feature dimensions
```bash
python train.py training.batch_size=2 model.backbone.d_model=64
```

### Issue: "Not enough samples for k-shot"
**Solution**: Use larger train_ratio or smaller k_shot
```bash
python train.py data.train_ratio=0.2 few_shot.k_shot=3
```

### Issue: "Key not found in .mat file"
**Solution**: Check the actual keys in your .mat file
```python
import scipy.io
data = scipy.io.loadmat('your_file.mat')
print(data.keys())
```

Update `image_key` and `gt_key` in the dataset config accordingly.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{hsi_fewshot_2024,
  title={Few-Shot Hyperspectral Image Classification Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/hsi-fewshot}
}
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Prototypical Networks: Snell et al., 2017
- 3D-CNN for HSI: Various works on spatial-spectral feature extraction
- Hydra: Facebook Research for configuration management

## Contact

For questions or issues, please open a GitHub issue or contact: your.email@example.com
