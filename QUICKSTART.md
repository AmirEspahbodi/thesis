# Quick Start Guide - HSI Few-Shot Classification

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
cd hsi_fewshot
pip install -r requirements.txt
```

### Step 2: Prepare Your Data
Place your hyperspectral .mat file in the `data/` directory:
```
hsi_fewshot/data/Indian_pines_corrected.mat
```

### Step 3: Run the Pipeline
```bash
# Basic run with defaults
python main.py

# Custom configuration
python main.py --data_path data/Pavia_University.mat --n_way 10 --k_shot 3 --num_epochs 100
```

## 📊 Expected Output

```
============================================================
HSI Few-Shot Classification Pipeline
============================================================
Experiment: hsi_fewshot_exp1
Device: cuda
Random Seed: 42

============================================================
Step 1: Data Loading and Preprocessing
============================================================
Loaded HSI data with shape: (145, 145, 200)
Loaded GT data with shape: (145, 145)
Number of classes: 16

PCA: Reduced 200 bands to 30 components
Explained variance: 0.9856

Extracted 10249 patches of size 9x9x30

============================================================
Step 2: Model Initialization
============================================================
Model created successfully!
  Total trainable parameters: 1,234,567
  Embedding dimension: 256

============================================================
Step 3: Training and Evaluation
============================================================
Epoch [1] Episode [50/1000] Loss: 1.2345 | Acc: 0.7234
...

Final Evaluation on Target Domain
==================================================
Classification Metrics:
==================================================
Overall Accuracy (OA): 0.8234 (82.34%)
Average Accuracy (AA): 0.7956 (79.56%)
Kappa Coefficient (κ): 0.7845
==================================================

Results saved to: results/hsi_fewshot_exp1_20250205_143022.json
```

## 🎯 Key Files to Know

- `main.py` - Run the complete pipeline
- `configs/config.py` - Configure hyperparameters
- `example_usage.py` - See code examples
- `test_framework.py` - Validate installation
- `README.md` - Full documentation

## 🔧 Common Configurations

### Small Test Run
```bash
python main.py --num_epochs 5 --n_way 3 --k_shot 3
```

### Large Scale Experiment
```bash
python main.py --n_way 10 --k_shot 5 --num_epochs 100 --device cuda
```

### Different Dataset
```python
# Edit configs/config.py
config.data.data_path = "data/Pavia_University.mat"
config.data.hsi_key = "pavia"
config.data.gt_key = "pavia_gt"
```

## 📁 Output Files

After running, you'll find:
- `checkpoints/best_model.pt` - Best model weights
- `results/*.json` - Experiment metrics
- `logs/*.log` - Training logs

## ⚡ Performance Tips

1. **GPU Usage**: Add `--device cuda` for faster training
2. **Batch Size**: Adjust `q_query` in config to fit GPU memory
3. **Episodes**: More training episodes = better convergence
4. **PCA Components**: 30-40 usually works well for HSI

## 🐛 Troubleshooting

**Import Errors?**
```bash
pip install torch numpy scipy scikit-learn
```

**CUDA Out of Memory?**
```python
config.fewshot.q_query = 10  # Reduce from 15
```

**File Not Found?**
- Check `data/` directory exists
- Verify .mat file path is correct
- Update `hsi_key` and `gt_key` in config

## 📚 Learn More

- See `README.md` for detailed documentation
- Check `example_usage.py` for code examples
- Read docstrings in source files for API details
- Run `test_framework.py` to validate setup

## ✅ Validation Checklist

- [ ] Dependencies installed
- [ ] Data file in place
- [ ] Config updated (if needed)
- [ ] GPU available (optional but recommended)
- [ ] Ran test_framework.py successfully

Ready to train! 🚀
