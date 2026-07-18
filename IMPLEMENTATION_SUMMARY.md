# Summary of Updates for IEEE Transmission Grid Evaluation

## What Was Done

### 1. **Models Added to models.py**
- ✅ **GIN** (Graph Isomorphism Network)
- ✅ **GAT** (Graph Attention Network) 
- ✅ **TransformerGNN** (Graph Transformer)
- Plus existing: GCN, ARMA_GNN

All models:
- Accept input_dim=7 for node features
- Output 4-dimensional predictions
- Handle edge attributes and NaN values
- Compatible with PyTorch Geometric DataLoader

### 2. **New Evaluation Scripts**

#### **eval_ieee_models.py** (Single Evaluation)
- Load pre-trained .pth files
- Evaluate one model on one test grid
- Compute NRMSE + MMD + g-score
- Save results to CSV
- Supports: IEEE18, IEEE24, IEEE39, UK grids

```bash
python eval_ieee_models.py \
  --data_dir ./ieee-grids \
  --training_grid IEEE18 \
  --test_grid IEEE24 \
  --model_path ./trained_models/ieee18_gcn.pth \
  --model_type gcn
```

#### **batch_eval_ieee.py** (Batch Evaluation)
- Train on 1 grid, test on remaining 3
- Evaluate multiple models
- Generate summary statistics
- Auto-search for model files

```bash
python batch_eval_ieee.py \
  --data_dir ./ieee-grids \
  --model_dir ./trained_models \
  --training_grid IEEE18 \
  --models gcn gin gat transformer
```

### 3. **Documentation Files**

| File | Purpose |
|------|---------|
| **IEEE_EVALUATION_GUIDE.md** | Complete reference guide |
| **IEEE_QUICK_START.md** | Quick reference card |
| **EXAMPLES_EVAL.py** | 10 practical examples |

## Key Features

### ✅ .pth File Support
```python
# Load state dictionary directly
state_dict = torch.load('ieee18_gcn.pth', weights_only=True)
model.load_state_dict(state_dict)
```

### ✅ IEEE Grid Support
- **IEEE18**: 18 bus system (small)
- **IEEE24**: 24 bus system (medium)
- **IEEE39**: 39 bus system (large)
- **UK**: Very large transmission system

### ✅ Batch Processing
- Automatically find model files (flexible naming)
- Process all test grids in one command
- Generate summary statistics
- Track progress with detailed output

### ✅ G-Score Calculation
```python
get_generalization_score(mmd_values, nrmse_values)
# Returns: mean_nrmse, std_nrmse, mmd_range, g_score
```

## Directory Structure

```
your-project/
├── models.py (updated)
│   └── +GIN, GAT, TransformerGNN
├── eval_ieee_models.py (new)
├── batch_eval_ieee.py (new)
├── IEEE_EVALUATION_GUIDE.md (new)
├── IEEE_QUICK_START.md (new)
├── EXAMPLES_EVAL.py (new)
├── ieee-grids/
│   ├── IEEE18/train/dataset.pt
│   ├── IEEE24/train/dataset.pt
│   ├── IEEE39/train/dataset.pt
│   └── UK/train/dataset.pt
└── trained_models/
    ├── ieee18_gcn.pth
    ├── ieee18_gin.pth
    ├── ieee18_gat.pth
    ├── ieee18_transformer.pth
    └── ...
```

## Quick Usage

### Single Evaluation
```bash
# Evaluate GCN trained on IEEE18, tested on IEEE24
python eval_ieee_models.py \
  --data_dir ./ieee-grids \
  --training_grid IEEE18 \
  --test_grid IEEE24 \
  --model_path ./trained_models/ieee18_gcn.pth \
  --model_type gcn
```

**Output**: `evaluation_results.csv`
```
training_grid,test_grid,model_type,nrmse_test,g_score_degree,...
IEEE18,IEEE24,gcn,0.125,0.142,...
```

### Batch Evaluation
```bash
# Evaluate all models (GCN, GIN, GAT, Transformer) 
# trained on IEEE18, tested on IEEE24, IEEE39, UK
python batch_eval_ieee.py \
  --data_dir ./ieee-grids \
  --model_dir ./trained_models \
  --training_grid IEEE18 \
  --models gcn gin gat transformer
```

**Output**:
- `batch_results_IEEE18.csv` (12 rows: 4 models × 3 test grids)
- `summary_IEEE18.csv` (4 rows: summary statistics per model)

## Model Comparison

| Model | Complexity | Speed | Best For |
|-------|-----------|-------|----------|
| GCN | Low | Fast ✓ | Baseline, transmission grids |
| GIN | Medium | Medium | Complex topology |
| GAT | Medium | Medium | Adaptive node interactions |
| Transformer | High | Slow | Highest accuracy |

## Workflow for Your Use Case

### Step 1: Prepare Data
```python
# Convert IEEE grids to PyTorch format
torch.save(dataset, 'ieee-grids/IEEE18/train/dataset.pt')
torch.save(dataset, 'ieee-grids/IEEE24/train/dataset.pt')
# ... repeat for IEEE39, UK
```

### Step 2: Train Models
```bash
# Train each model on IEEE18
for model in gcn gin gat transformer:
    python train_model.py --grid IEEE18 --model $model
    # Saves to: trained_models/ieee18_{model}.pth
done
```

### Step 3: Evaluate
```bash
# Batch evaluate all models
python batch_eval_ieee.py \
  --data_dir ./ieee-grids \
  --model_dir ./trained_models \
  --training_grid IEEE18 \
  --models gcn gin gat transformer \
  --output_dir ./results_ieee18
```

### Step 4: Analyze Results
```python
import pandas as pd

# Load results
results = pd.read_csv('results_ieee18/batch_results_IEEE18.csv')

# Best model
best = results.loc[results['g_score_degree'].idxmin()]
print(f"Best: {best['model_type']} on {best['test_grid']}")

# Average by model
print(results.groupby('model_type')['g_score_degree'].mean().sort_values())
```

## Output Interpretation

### G-Score
- **Lower = Better** (0.05-0.10 = excellent, >0.5 = challenging)
- Combines NRMSE (accuracy) + MMD (domain shift)
- Lower score = better generalization to new grid

### NRMSE
- **Normalized Root Mean Square Error**
- Typical range: 0.01-0.3
- Lower = more accurate predictions

### MMD
- **Maximum Mean Discrepancy** (graph structure difference)
- Degree: Node degree distribution difference
- Laplacian: Eigenvalue spectrum difference
- Higher MMD = more structurally different grids

## Features Supported

### Graph Augmentations
```bash
--add_cycles       # Include cycle length features
--add_path_lengths # Include path-to-slack features  
--add_degree       # Include node degree features
```

### Configuration
```bash
--batch_size 16              # Batch size
--sigma_degree 1e2           # MMD kernel parameter (degree)
--sigma_laplacian 1e-2       # MMD kernel parameter (Laplacian)
--verbose                    # Print detailed logs
```

## Files Modified/Created

### Modified
- ✅ **models.py** - Added GIN, GAT, TransformerGNN

### Created
- ✅ **eval_ieee_models.py** - Single evaluation script (200+ lines)
- ✅ **batch_eval_ieee.py** - Batch evaluation script (300+ lines)
- ✅ **IEEE_EVALUATION_GUIDE.md** - Complete reference (~400 lines)
- ✅ **IEEE_QUICK_START.md** - Quick reference (~200 lines)
- ✅ **EXAMPLES_EVAL.py** - 10 practical examples (~400 lines)

## Next Steps

1. **Prepare your data** - Convert IEEE grids to PyTorch format
2. **Train models** - On one grid (e.g., IEEE18)
3. **Run evaluation** - `batch_eval_ieee.py` on remaining grids
4. **Analyze results** - Compare model generalization
5. **Iterate** - Try different architectures, hyperparameters

## Expected Behavior

```
Training grid: IEEE18
Test grids: IEEE24, IEEE39, UK

Expected g-scores:
- IEEE18 → IEEE24: 0.12-0.18 (similar size, lower score = better)
- IEEE18 → IEEE39: 0.15-0.25 (larger grid)
- IEEE18 → UK: 0.25-0.50+ (very different size/structure)

Best model typically: Transformer > GAT > GIN > GCN
But transmission grids often favor: GCN or GAT (faster, stable)
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found | Check pattern: `{lowercase_grid}_{model_type}.pth` |
| Shape mismatch | Ensure x has shape (N, 7) for all grids |
| Out of memory | Reduce batch_size or test fewer grids |
| High g-score | Normal for very different grids (domain shift) |
| NaN results | Check for unexpected NaNs in input data |

## Support

- See **IEEE_EVALUATION_GUIDE.md** for detailed reference
- See **IEEE_QUICK_START.md** for quick commands
- See **EXAMPLES_EVAL.py** for code examples
- See **CUSTOM_DATASET_GUIDE.md** for data format requirements

---

**Ready to evaluate?** Start with:
```bash
python eval_ieee_models.py --help
python batch_eval_ieee.py --help
```
