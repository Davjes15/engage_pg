# Evaluating Custom Trained Models and Computing G-Score

## Overview

To evaluate your pre-trained model and compute the g-score, you need to:

1. **Prepare your data** in the required format (see CUSTOM_DATASET_GUIDE.md)
2. **Load your trained model**
3. **Run inference** on the test dataset to get NRMSE
4. **Calculate MMD** between training and test distributions
5. **Compute g-score** combining NRMSE and MMD metrics

## Data Preparation

Organize your datasets as:

```
your-data-dir/
├── training-grid/
│   └── train/
│       ├── dataset.pt
│       └── dataset_src.csv
└── test-grid/
    └── train/
        ├── dataset.pt
        └── dataset_src.csv
```

**Important**: Use `train/` folder for both training and test data (the code loads from `train/` by default).

## Step-by-Step Process

### Option 1: Using Existing Experiment Script

If your model was trained using the repo's training utilities, you can extend the experiment scripts:

```bash
python cross_context_experiment.py \
  --data_dir your-data-dir/ \
  --model gcn \
  --load_model_dir path/to/trained/model \
  --eval_only \
  --save_results \
  --mmd
```

**Parameters**:
- `--load_model_dir`: Directory containing your trained `model_weights_0` file
- `--eval_only`: Skip training, only evaluate
- `--save_results`: Save results to CSV

**Output**: Results CSV with g-score calculations

### Option 2: Custom Evaluation Script (Recommended for Your Use Case)

Create a custom script for more control:

```python
# eval_custom_model.py

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from models import GCN, ARMA_GNN
from training_utils import (
    get_dataloaders,
    get_device,
    test,
    evaluate_mmd,
    get_generalization_score
)
from graph_utils import get_networkx_graph, get_pyg_graphs

# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = 'path/to/your-data-dir/'
TRAINING_GRID = 'training-grid'          # Your training grid name
TEST_GRID = 'test-grid'                   # Your test grid name
MODEL_PATH = 'path/to/trained/model'     # Path to saved model weights
MODEL_TYPE = 'gcn'                        # 'gcn' or 'arma_gnn'

# ============================================================================
# Load Model
# ============================================================================

device = get_device()
print(f"Using device: {device}")

# Load training data to get input dimension
train_loader, _, _ = get_dataloaders(
    data_dir=DATA_DIR,
    training_grids=[TRAINING_GRID],
    testing_grids=None,
    batch_size=16
)

input_dim = next(iter(train_loader)).x.shape[1]
print(f"Input dimension: {input_dim}")

# Create model
model_classes = {'gcn': GCN, 'arma_gnn': ARMA_GNN}
model = model_classes[MODEL_TYPE](input_dim=input_dim).to(device)

# Load trained weights
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location=device))
print(f"Model loaded from: {MODEL_PATH}")

# ============================================================================
# Evaluate on Test Data
# ============================================================================

_, _, test_loader = get_dataloaders(
    data_dir=DATA_DIR,
    training_grids=None,
    testing_grids=[TEST_GRID],
    batch_size=16
)

print("\nEvaluating on test set...")
nrmse_test = test(model=model, device=device, loader_test=test_loader)
print(f"Test NRMSE: {nrmse_test:.6f}")

# ============================================================================
# Calculate MMD (Graph Dissimilarity)
# ============================================================================

print("\nCalculating MMD between training and test graphs...")

# Load graph data
training_graphs = get_pyg_graphs(DATA_DIR, TRAINING_GRID, split='train')
test_graphs = get_pyg_graphs(DATA_DIR, TEST_GRID, split='train')

# Evaluate MMD
mmd_degree, mmd_laplacian = evaluate_mmd(
    training_dataset=training_graphs,
    testing_dataset=test_graphs,
    sigma_degree=1e2,
    sigma_laplacian=1e-2
)

print(f"MMD Degree Distribution: {mmd_degree:.6f}")
print(f"MMD Laplacian Spectrum: {mmd_laplacian:.6f}")

# ============================================================================
# Calculate G-Score
# ============================================================================

print("\nCalculating Generalization Score...")

# For a single evaluation pair, we need arrays of values
# Repeat the values to create a minimal dataset for g-score calculation
nrmses = np.array([nrmse_test])
mmds_degree = np.array([mmd_degree])
mmds_laplacian = np.array([mmd_laplacian])

# Calculate g-score components
mean_nrmse, std_nrmse, mmd_range_degree, g_score_degree = get_generalization_score(
    mmds_degree, nrmses
)

_, _, mmd_range_laplacian, g_score_laplacian = get_generalization_score(
    mmds_laplacian, nrmses
)

# ============================================================================
# Results Summary
# ============================================================================

print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)
print(f"Training Grid: {TRAINING_GRID}")
print(f"Test Grid: {TEST_GRID}")
print(f"Model Type: {MODEL_TYPE}")
print("-"*60)
print(f"Test NRMSE: {nrmse_test:.6f}")
print(f"MMD Degree Distribution: {mmd_degree:.6f}")
print(f"MMD Laplacian Spectrum: {mmd_laplacian:.6f}")
print("-"*60)
print(f"Mean NRMSE: {mean_nrmse:.6f}")
print(f"Std NRMSE: {std_nrmse:.6f}")
print(f"MMD Range (Degree): {mmd_range_degree:.6f}")
print(f"MMD Range (Laplacian): {mmd_range_laplacian:.6f}")
print("-"*60)
print(f"G-Score (Degree): {g_score_degree:.6f}")
print(f"G-Score (Laplacian): {g_score_laplacian:.6f}")
print("="*60)

# Save results to CSV
results_df = pd.DataFrame({
    'training_grid': [TRAINING_GRID],
    'test_grid': [TEST_GRID],
    'model_type': [MODEL_TYPE],
    'nrmse_test': [nrmse_test],
    'mmd_degree': [mmd_degree],
    'mmd_laplacian': [mmd_laplacian],
    'mean_nrmse': [mean_nrmse],
    'std_nrmse': [std_nrmse],
    'mmd_range_degree': [mmd_range_degree],
    'mmd_range_laplacian': [mmd_range_laplacian],
    'g_score_degree': [g_score_degree],
    'g_score_laplacian': [g_score_laplacian]
})

output_file = 'evaluation_results.csv'
results_df.to_csv(output_file, index=False)
print(f"\nResults saved to: {output_file}")
```

## Usage

### Step 1: Prepare Your Data

Format both training and test datasets according to CUSTOM_DATASET_GUIDE.md:

```
my-data/
├── transmission-grid-1/
│   └── train/
│       ├── dataset.pt
│       └── dataset_src.csv
└── transmission-grid-2/
    └── train/
        ├── dataset.pt
        └── dataset_src.csv
```

### Step 2: Modify the Configuration

Edit the script's configuration section:

```python
DATA_DIR = 'my-data/'
TRAINING_GRID = 'transmission-grid-1'
TEST_GRID = 'transmission-grid-2'
MODEL_PATH = 'path/to/trained/model_weights'
MODEL_TYPE = 'gcn'  # or 'arma_gnn'
```

### Step 3: Run the Script

```bash
python eval_custom_model.py
```

### Step 4: Interpret Results

The script outputs:

- **NRMSE_test**: Normalized Root Mean Square Error on test data
  - Lower is better
  - Range: [0, ∞), typically 0.01-0.3

- **MMD Metrics**: How different train/test graphs are structurally
  - Degree Distribution: Based on node degree
  - Laplacian Spectrum: Based on global graph structure
  - Higher MMD = more dissimilar graphs

- **G-Score**: Combined metric balancing error and graph dissimilarity
  - Lower is better
  - Formula: `mean_nrmse + std_nrmse * log(mmd_range + 1) / (mmd_range + eps)`
  - Accounts for prediction reliability across graph diversity

## Understanding G-Score Components

### Mean NRMSE
- Average prediction error over your test dataset
- Primary indicator of model performance

### Std NRMSE
- Variability in predictions
- High std = inconsistent performance across samples

### MMD Range
- For this single-pair evaluation, range is 0 (only one value)
- In multi-pair scenarios, represents spread of MMD values
- **Note**: For single evaluation, this metric is less meaningful

## For Multiple Test Grids

If you want to evaluate on multiple test grids and get a more robust g-score:

```python
# eval_multiple_grids.py

test_grids = ['grid-1', 'grid-2', 'grid-3', ...]
results = []

for test_grid in test_grids:
    # Load test data
    _, _, test_loader = get_dataloaders(
        data_dir=DATA_DIR,
        training_grids=None,
        testing_grids=[test_grid],
        batch_size=16
    )
    
    # Evaluate
    nrmse = test(model=model, device=device, loader_test=test_loader)
    
    # Calculate MMD
    training_graphs = get_pyg_graphs(DATA_DIR, TRAINING_GRID, split='train')
    test_graphs = get_pyg_graphs(DATA_DIR, test_grid, split='train')
    mmd_degree, mmd_laplacian = evaluate_mmd(training_graphs, test_graphs)
    
    results.append({
        'test_grid': test_grid,
        'nrmse': nrmse,
        'mmd_degree': mmd_degree,
        'mmd_laplacian': mmd_laplacian
    })

# Calculate g-score with all results
nrmses = np.array([r['nrmse'] for r in results])
mmds = np.array([r['mmd_degree'] for r in results])

mean_nrmse, std_nrmse, mmd_range, g_score = get_generalization_score(mmds, nrmses)

print(f"G-Score across {len(test_grids)} test grids: {g_score:.6f}")
```

## Transmission Grid Specific Considerations

For transmission grids specifically:

1. **Larger networks**: May have more nodes/edges than distribution grids
   - May require more GPU memory
   - Consider reducing batch_size if needed

2. **Voltage levels**: Likely 100+ kV
   - Ensure features are properly normalized
   - DC-PF approximations are typically good for transmission

3. **Sparse connectivity**: Transmission grids are typically more sparse
   - Graph structure differences (MMD) may be larger
   - G-score may reflect significant domain shift

4. **Feature scaling**: Verify power values are in consistent units
   - Use per-unit (p.u.) representation
   - Ensure impedance values are normalized

## Troubleshooting

### "Shape mismatch" error
- Ensure test dataset has same feature dimensions as training (x must be (N, 7))

### Model outputs NaN
- Check input data has no unexpected NaNs outside masked features

### Very high g-score
- High MMD indicates very different grid structures
- This is expected when training on distribution grids and testing on transmission grids

### MMD calculation fails
- Ensure graphs are valid NetworkX objects
- Check that both datasets have non-empty graphs

## Alternative: One-to-One Comparison

If you only have one trained model and one test grid, your evaluation is straightforward:

```
Training Distribution (Grid A) → Train Model
                                    ↓
                            Evaluate on Test Grid B
                                    ↓
                            NRMSE + MMD(A,B) + G-Score
```

The g-score in this case tells you: "How well does the model generalize from Grid A to Grid B?"

Lower g-score = better cross-grid generalization
