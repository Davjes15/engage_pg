# Quick Reference: Evaluating Your Transmission Grid Model

## Your Scenario

- **Trained on**: Your transmission grid dataset
- **Test on**: Another proprietary transmission grid dataset  
- **Goal**: Get model performance + g-score

## Preparation Checklist

- [ ] Both datasets formatted per CUSTOM_DATASET_GUIDE.md
- [ ] Directory structure: `data-dir/grid-name/train/dataset.pt`
- [ ] Trained model weights saved (e.g., `model_weights.pt`)
- [ ] Both grids have multiple samples for meaningful MMD calculation

## One-Line Quick Start

```bash
python eval_trained_model.py \
  --data_dir ./my-grids \
  --training_grid "transmission-grid-1" \
  --test_grid "transmission-grid-2" \
  --model_path ./trained_model.pt \
  --model_type gcn
```

## What Gets Calculated

| Metric | What It Means | Good Range |
|--------|--------------|-----------|
| **NRMSE** | Prediction error on test set | 0.01 - 0.3 |
| **MMD_degree** | Difference in node degree distributions | Variable |
| **MMD_laplacian** | Difference in graph spectra | Variable |
| **Mean NRMSE** | Average error (same as NRMSE for single eval) | Lower = better |
| **Std NRMSE** | Prediction variability | Lower = better |
| **G-Score** | Combined metric: error + reliability | Lower = better |

## Understanding Your Output

```
Test NRMSE: 0.125          ← Model accuracy on test grid
MMD Degree: 45.3           ← How different the grids are (node degrees)
MMD Laplacian: 0.082       ← How different the grids are (structure)
G-SCORE: 0.142             ← Overall generalization quality
```

**Interpretation**:
- **Low NRMSE + Low G-Score** = Model generalizes well to the new grid
- **High NRMSE + High G-Score** = Model struggles (domain shift or training issues)
- **Low NRMSE + High G-Score** = Model performs okay but unreliable
- **High MMD** = New grid structure differs significantly from training grid

## With Feature Augmentations

If your model was trained with additional features (cycles, path lengths, degree):

```bash
python eval_trained_model.py \
  --data_dir ./my-grids \
  --training_grid "transmission-grid-1" \
  --test_grid "transmission-grid-2" \
  --model_path ./trained_model.pt \
  --model_type gcn \
  --add_cycles \
  --add_path_lengths \
  --add_degree
```

**Important**: Use the same augmentations as your trained model!

## Multiple Test Grids

Test on multiple grids to get robust g-score:

```bash
# Grid B
python eval_trained_model.py \
  --data_dir ./my-grids \
  --training_grid "transmission-grid-1" \
  --test_grid "transmission-grid-2" \
  --model_path ./trained_model.pt \
  --output_file results_grid2.csv

# Grid C
python eval_trained_model.py \
  --data_dir ./my-grids \
  --training_grid "transmission-grid-1" \
  --test_grid "transmission-grid-3" \
  --model_path ./trained_model.pt \
  --output_file results_grid3.csv
```

Then combine results and recalculate g-score:

```python
import pandas as pd
import numpy as np
from training_utils import get_generalization_score

# Load all results
results = pd.concat([
    pd.read_csv('results_grid2.csv'),
    pd.read_csv('results_grid3.csv'),
    # ... more grids
])

# Calculate g-score across all grids
nrmses = results['nrmse_test'].values
mmds = results['mmd_degree'].values  # or mmd_laplacian

mean_nrmse, std_nrmse, mmd_range, g_score = get_generalization_score(mmds, nrmses)

print(f"Overall G-Score: {g_score:.6f}")
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Module not found" | Run from repo root directory |
| "Shape mismatch" | Ensure x features are (N, 7) - check CUSTOM_DATASET_GUIDE.md |
| Very high NRMSE (>1.0) | Grid structures very different, or model not trained well |
| Model outputs NaN | Check for unexpected NaNs in input data |
| "File not found" | Verify paths to model weights and data directories |

## Output Files

The script creates a CSV with columns:

```
training_grid, test_grid, model_type, nrmse_test, 
mmd_degree, mmd_laplacian, mean_nrmse, std_nrmse,
mmd_range_degree, mmd_range_laplacian,
g_score_degree, g_score_laplacian
```

## Manual Step-by-Step (if you need more control)

See `EVALUATE_CUSTOM_MODEL.md` for detailed Python code to:
- Load model manually
- Evaluate specific samples
- Calculate metrics independently
- Analyze per-grid or per-sample results

## Key Formulas

**G-Score**:
```
g_score = mean_nrmse + std_nrmse × log(mmd_range + 1) / (mmd_range + ε)
```

**NRMSE** (Normalized Root Mean Squared Error):
```
nrmse = √(MSE) / avg_feature_range
```

**MMD** (Maximum Mean Discrepancy):
```
Kernel-based distance between two graph distributions
- Degree: Based on node degree distributions
- Laplacian: Based on normalized Laplacian spectra
```

## Next Steps

1. ✓ Prepare both datasets (see CUSTOM_DATASET_GUIDE.md)
2. ✓ Run evaluation script
3. → Analyze g-score and MMD to understand generalization
4. → If g-score is high, consider:
   - Training with more diverse grids
   - Adding graph structure features (cycles, paths)
   - Using different model architecture
   - Using domain adaptation techniques
