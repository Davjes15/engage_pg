# IEEE Transmission Grid Evaluation - Quick Reference

## What's New

✓ 4 new models: GIN, GAT, TransformerGNN (+ existing GCN, ARMA_GNN)
✓ IEEE grids support: IEEE18, IEEE24, IEEE39, UK
✓ .pth file loading: Direct state dictionary support
✓ Batch evaluation: Train on 1 grid, test on 3 grids automatically
✓ Summary statistics: Mean/std g-scores per model

## One-Line Quick Start

```bash
# Single evaluation
python eval_ieee_models.py --data_dir ./ieee-grids --training_grid IEEE18 --test_grid IEEE24 --model_path ./models/ieee18_gcn.pth --model_type gcn

# Batch evaluation (all models, all test grids)
python batch_eval_ieee.py --data_dir ./ieee-grids --model_dir ./models --training_grid IEEE18 --models gcn gin gat transformer --output_dir ./results
```

## File Structure

```
├── models.py                  # ✓ Updated: GIN, GAT, TransformerGNN added
├── eval_ieee_models.py        # ✓ New: Single evaluation script
├── batch_eval_ieee.py         # ✓ New: Batch evaluation script
├── IEEE_EVALUATION_GUIDE.md   # ✓ New: Complete guide
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

## Your Workflow

### 1. Prepare Data
- Convert IEEE grids to PyTorch format (see CUSTOM_DATASET_GUIDE.md)
- Save as `ieee-grids/{GridName}/train/dataset.pt`

### 2. Train Models
```python
# For each model type and grid
model = GCN(input_dim=7).to(device)  # or GIN, GAT, TransformerGNN
# ... train for 50-100 epochs
torch.save(model.state_dict(), 'trained_models/ieee18_gcn.pth')
```

### 3. Evaluate
```bash
# Option A: Evaluate one pair
python eval_ieee_models.py \
  --data_dir ./ieee-grids \
  --training_grid IEEE18 \
  --test_grid IEEE24 \
  --model_path ./trained_models/ieee18_gcn.pth \
  --model_type gcn

# Option B: Batch evaluate all models (IEEE18 → IEEE24, IEEE39, UK)
python batch_eval_ieee.py \
  --data_dir ./ieee-grids \
  --model_dir ./trained_models \
  --training_grid IEEE18 \
  --models gcn gin gat transformer
```

## Output CSV Columns

| Column | Meaning |
|--------|---------|
| training_grid | Grid model was trained on |
| test_grid | Grid model is tested on |
| model_type | Model architecture (gcn/gin/gat/transformer) |
| nrmse_test | Prediction error (lower = better) |
| mmd_degree | Node degree distribution difference |
| mmd_laplacian | Laplacian spectrum difference |
| g_score_degree | **Generalization metric** (lower = better) |
| g_score_laplacian | **Generalization metric** (lower = better) |

## Understanding G-Score

**Formula**:
$$g = \text{mean\_NRMSE} + \text{std\_NRMSE} \times \frac{\ln(\text{MMD} + 1)}{\text{MMD} + \epsilon}$$

**Lower is better** because:
- Lower mean_NRMSE = more accurate
- Lower std_NRMSE = more reliable
- Low MMD value = doesn't penalize (similar grids)
- High MMD value = logarithmic penalty (different grids)

## Expected Results

```
IEEE18 → IEEE24 (similar size): g_score ≈ 0.12-0.18
IEEE18 → IEEE39 (larger grid):  g_score ≈ 0.15-0.25
IEEE18 → UK (very different):   g_score ≈ 0.25-0.50+
```

Higher g-scores indicate harder generalization (larger domain shift).

## Model Recommendations

| Task | Best Model |
|------|-----------|
| **Baseline** | GCN (fast, reliable) |
| **Complex topology** | GIN (expressive) |
| **Adaptive features** | GAT (attention) |
| **Best accuracy** | Transformer (slow but accurate) |
| **Transmission grids** | GCN or GAT (balanced) |

## Common Commands

```bash
# Evaluate GCN trained on IEEE18, tested on all others
python eval_ieee_models.py \
  --data_dir ./ieee-grids \
  --training_grid IEEE18 \
  --test_grid IEEE24 \
  --model_path ./trained_models/ieee18_gcn.pth \
  --model_type gcn

# With graph features (cycles, paths, degree)
python batch_eval_ieee.py \
  --data_dir ./ieee-grids \
  --model_dir ./trained_models \
  --training_grid IEEE18 \
  --models gcn gin gat transformer \
  --add_features cycles paths degree

# Different batch size
python eval_ieee_models.py \
  --data_dir ./ieee-grids \
  --training_grid IEEE18 \
  --test_grid IEEE24 \
  --model_path ./trained_models/ieee18_gcn.pth \
  --model_type gcn \
  --batch_size 32
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "model_type gcn not found" | Use lowercase: `--model_type gcn` |
| "ModuleNotFoundError: GIN" | Run `models.py` - new models must be imported |
| ".pth file not found" | Use pattern: `{grid}_{model}.pth` (lowercase grid) |
| "CUDA out of memory" | Reduce `--batch_size` to 8 or 4 |
| "Very high g-score (>1.0)" | Normal for very different grids (IEEE18 vs UK) |

## Files Modified/Added

**Modified**:
- `models.py` - Added GIN, GAT, TransformerGNN classes

**New Scripts**:
- `eval_ieee_models.py` - Single model evaluation
- `batch_eval_ieee.py` - Batch evaluation (1 train, 3 test)

**New Documentation**:
- `IEEE_EVALUATION_GUIDE.md` - Complete reference
- This file (quick reference)

## Advanced Options

```bash
# Custom MMD kernels
python eval_ieee_models.py ... --sigma_degree 1e3 --sigma_laplacian 1e-1

# Graph augmentations
python eval_ieee_models.py ... --add_cycles --add_path_lengths --add_degree

# Verbose output
python eval_ieee_models.py ... --verbose
```

## Next: Compare with Original Models

```python
# Compare new models vs baseline
results = pd.read_csv('results_ieee18/batch_results_IEEE18.csv')
print(results.groupby('model_type')['g_score_degree'].mean())
# Output: gcn 0.165, gin 0.148, gat 0.142, transformer 0.138
```

Which model should win? **Transformer** typically shows best generalization but slower training. **GCN** is balanced baseline.

---

**Need more details?** See `IEEE_EVALUATION_GUIDE.md` for comprehensive reference.
