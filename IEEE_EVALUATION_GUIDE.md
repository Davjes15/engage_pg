# IEEE Transmission Grid Model Evaluation Guide

## Overview

You can now evaluate your pre-trained models (GCN, GIN, GAT, Transformer) on IEEE transmission grids. The framework supports:

- **Multiple model architectures**: GCN, GIN, GAT, TransformerGNN, ARMA_GNN
- **IEEE grids**: IEEE18, IEEE24, IEEE39, UK
- **Train-test strategy**: Train on one grid, test on remaining three
- **.pth file support**: Direct loading of PyTorch state dictionaries

## New Models Added

### 1. GIN (Graph Isomorphism Network)
```python
model = GIN(input_dim=7, num_layers=8)
```
- Powerful for graph structure capture
- Uses learnable MLP aggregators
- Good for power grid topology analysis

### 2. GAT (Graph Attention Network)
```python
model = GAT(input_dim=7, num_layers=8, num_heads=4)
```
- Attention-based node interaction
- Multi-head attention (4 heads default)
- Adaptive edge importance weighting

### 3. TransformerGNN
```python
model = TransformerGNN(input_dim=7, num_layers=8)
```
- State-of-the-art graph transformers
- Global attention mechanism
- Excellent for long-range dependencies

## Setup

### Directory Structure

```
your-project/
├── ieee-grids/
│   ├── IEEE18/
│   │   └── train/
│   │       ├── dataset.pt
│   │       └── dataset_src.csv
│   ├── IEEE24/
│   │   └── train/
│   │       ├── dataset.pt
│   │       └── dataset_src.csv
│   ├── IEEE39/
│   │   └── train/
│   │       ├── dataset.pt
│   │       └── dataset_src.csv
│   └── UK/
│       └── train/
│           ├── dataset.pt
│           └── dataset_src.csv
├── trained_models/
│   ├── ieee18_gcn.pth
│   ├── ieee18_gin.pth
│   ├── ieee18_gat.pth
│   ├── ieee18_transformer.pth
│   ├── ieee24_gcn.pth
│   └── ...
```

### Data Format

Each dataset must follow the PyTorch Geometric format (see CUSTOM_DATASET_GUIDE.md):

```python
# Each dataset.pt contains a list of Data objects
[
    Data(
        x=(N1, 7),           # Node features
        edge_index=(2, 2E1), # Edges
        edge_attr=(2E1, 4),  # Edge features
        y=(N1, 4),           # Ground truth
        dc_pf=(N1, 4)        # DC power flow
    ),
    Data(...),
    ...
]
```

### Model Weights Format

Save models with complete state dictionaries:

```python
import torch

# After training
torch.save(model.state_dict(), 'ieee18_gcn.pth')

# Loading (handled automatically by eval scripts)
state_dict = torch.load('ieee18_gcn.pth', weights_only=True)
model.load_state_dict(state_dict)
```

## Usage

### Single Evaluation

Evaluate one model on one test grid:

```bash
python eval_ieee_models.py \
  --data_dir ./ieee-grids \
  --training_grid IEEE18 \
  --test_grid IEEE24 \
  --model_path ./trained_models/ieee18_gcn.pth \
  --model_type gcn \
  --output_file results_ieee18_gcn.csv
```

**Optional flags**:
```bash
--add_cycles            # Include cycle length features
--add_path_lengths      # Include path length features
--add_degree            # Include node degree features
--batch_size 32         # Batch size (default: 16)
--sigma_degree 1e2      # MMD degree kernel sigma
--sigma_laplacian 1e-2  # MMD Laplacian kernel sigma
--verbose               # Print detailed logs
```

### Batch Evaluation (One Training Grid, Three Test Grids)

Train on IEEE18, test on IEEE24, IEEE39, UK:

```bash
python batch_eval_ieee.py \
  --data_dir ./ieee-grids \
  --model_dir ./trained_models \
  --training_grid IEEE18 \
  --models gcn gin gat transformer \
  --output_dir ./results_ieee18 \
  --batch_size 16
```

This will:
1. Test GCN trained on IEEE18 against IEEE24, IEEE39, UK
2. Test GIN trained on IEEE18 against IEEE24, IEEE39, UK
3. Test GAT trained on IEEE18 against IEEE24, IEEE39, UK
4. Test Transformer trained on IEEE18 against IEEE24, IEEE39, UK
5. Generate summary statistics for each model

**Output files**:
- `batch_results_IEEE18.csv` - Detailed results for all evaluations
- `summary_IEEE18.csv` - Summary statistics by model

### Model Naming Convention

The batch script automatically searches for model weights. It looks for files matching:
- `{training_grid}_{model_type}.pth` (e.g., `ieee18_gcn.pth`)
- `{model_type}_{training_grid}.pth` (e.g., `gcn_ieee18.pth`)

Use lowercase for grid names in filenames (IEEE18 → ieee18).

## Output Interpretation

### Single Evaluation Output

```csv
training_grid,test_grid,nrmse_test,mmd_degree,mmd_laplacian,mean_nrmse,std_nrmse,mmd_range_degree,mmd_range_laplacian,g_score_degree,g_score_laplacian
IEEE18,IEEE24,0.125,45.3,0.082,0.125,0.032,0.0,0.0,0.142,0.128
```

| Column | Meaning | Interpretation |
|--------|---------|-----------------|
| **nrmse_test** | Normalized RMSE on test grid | Lower = better, 0.01-0.3 typical |
| **mmd_degree** | Node degree distribution difference | Higher = more structural difference |
| **mmd_laplacian** | Graph spectrum difference | Higher = more structural difference |
| **g_score_degree** | Overall generalization metric (degree) | **Lower = better** |
| **g_score_laplacian** | Overall generalization metric (Laplacian) | **Lower = better** |

### Batch Evaluation Summary

```csv
model_type,nrmse_test_mean,nrmse_test_std,g_score_degree_mean,g_score_degree_std,...
gcn,0.145,0.032,0.165,0.042,...
gin,0.132,0.028,0.148,0.038,...
gat,0.128,0.025,0.142,0.035,...
transformer,0.125,0.022,0.138,0.031,...
```

## Example Workflow

### Step 1: Prepare IEEE Grids

Convert your IEEE transmission grids to PyTorch format:

```python
# prepare_ieee_grids.py
import torch
from torch_geometric.data import Data
import pandas as pd

def prepare_grid_dataset(grid_data, grid_name='IEEE18'):
    """Convert grid data to PyTorch Geometric format"""
    dataset = []
    
    for sample in grid_data:
        x = torch.tensor(sample['node_features'], dtype=torch.float32)  # (N, 7)
        y = torch.tensor(sample['ground_truth'], dtype=torch.float32)   # (N, 4)
        edge_index = torch.tensor(sample['edges'], dtype=torch.int64)   # (2, 2E)
        edge_attr = torch.tensor(sample['edge_features'], dtype=torch.float32)  # (2E, 4)
        dc_pf = torch.tensor(sample['dc_pf'], dtype=torch.float32)      # (N, 4)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, dc_pf=dc_pf)
        dataset.append(data)
    
    # Save dataset
    torch.save(dataset, f'ieee-grids/{grid_name}/train/dataset.pt')
    pd.DataFrame(['source.json'], columns=['src']).to_csv(
        f'ieee-grids/{grid_name}/train/dataset_src.csv'
    )

# Prepare all grids
prepare_grid_dataset(ieee18_data, 'IEEE18')
prepare_grid_dataset(ieee24_data, 'IEEE24')
prepare_grid_dataset(ieee39_data, 'IEEE39')
prepare_grid_dataset(uk_data, 'UK')
```

### Step 2: Train Models

Train each model on one grid (example: IEEE18):

```python
# train_ieee_model.py
from training_utils import train, get_dataloaders, get_device
from models import GCN, GIN, GAT, TransformerGNN
import torch

device = get_device()

for model_type in ['gcn', 'gin', 'gat', 'transformer']:
    print(f"Training {model_type.upper()}...")
    
    # Load data
    train_loader, val_loader, _ = get_dataloaders(
        data_dir='./ieee-grids',
        training_grids=['IEEE18'],
        batch_size=16
    )
    
    # Create model
    input_dim = next(iter(train_loader)).x.shape[1]
    model_classes = {'gcn': GCN, 'gin': GIN, 'gat': GAT, 'transformer': TransformerGNN}
    model = model_classes[model_type](input_dim=input_dim).to(device)
    
    # Train
    train_loss, val_loss, best_loss, train_time, epochs = train(
        model=model,
        device=device,
        loader_train=train_loader,
        loader_val=val_loader,
        epochs=100,
        learning_rate=1e-3,
        early_stopping=True,
        patience=50
    )
    
    # Save model
    torch.save(model.state_dict(), f'trained_models/ieee18_{model_type}.pth')
    print(f"Saved: trained_models/ieee18_{model_type}.pth")
```

### Step 3: Evaluate

```bash
# Evaluate all models trained on IEEE18
python batch_eval_ieee.py \
  --data_dir ./ieee-grids \
  --model_dir ./trained_models \
  --training_grid IEEE18 \
  --models gcn gin gat transformer \
  --output_dir ./results_ieee18
```

### Step 4: Analyze Results

```python
# analyze_results.py
import pandas as pd

results = pd.read_csv('results_ieee18/batch_results_IEEE18.csv')

# Best performing model on each test grid
best_by_grid = results.loc[results.groupby('test_grid')['g_score_degree'].idxmin()]
print(best_by_grid[['test_grid', 'model_type', 'g_score_degree']])

# Average generalization across all test grids
avg_by_model = results.groupby('model_type')['g_score_degree'].mean()
print(avg_by_model.sort_values())
```

## Model Comparison

Quick reference for model characteristics:

| Model | Parameters | Speed | Expressiveness | Best For |
|-------|-----------|-------|-----------------|----------|
| **GCN** | ~50K | Fast | Medium | Baseline |
| **GIN** | ~50K | Medium | High | Complex topology |
| **GAT** | ~60K | Medium | High | Adaptive features |
| **Transformer** | ~70K | Slow | Very High | Long-range deps |

## Transmission Grid Specifics

### Why Different Results by Grid?

- **IEEE18**: Small transmission system (18 buses)
- **IEEE24**: Medium system (24 buses)
- **IEEE39**: Larger system (39 buses)
- **UK**: Very large system (~1000+ buses) - may have different characteristics

Higher MMD values between grids = larger domain shift = harder generalization.

### Expected G-Score Ranges

- **Same grid** (IEEE18 → IEEE18): 0.05-0.10
- **Similar size grids** (IEEE18 → IEEE24): 0.10-0.20
- **Different sizes** (IEEE18 → IEEE39): 0.15-0.30
- **Very different** (IEEE18 → UK): 0.25-0.50+

If g-score is unexpectedly high, check:
1. Are datasets properly formatted?
2. Is MMD calculation running correctly?
3. Are models properly trained?

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "ModuleNotFoundError" | Ensure models.py has GIN, GAT, TransformerGNN defined |
| ".pth file not found" | Check filename matches pattern (lowercase grid names) |
| "Shape mismatch in model" | Ensure all grids have x shape (N, 7) |
| "NaN g-score" | Check for NaNs in MMD or NRMSE calculations |
| "Out of memory" | Reduce batch_size, use smaller test grid |
| "MMD calculation slow" | Reduce number of graphs or use smaller sigma values |

## Advanced: Custom Model Configuration

For finer control, edit the model initialization in eval_ieee_models.py:

```python
# Customize model architecture
model_configs = {
    'gcn': {'input_dim': 7, 'num_layers': 12},  # More layers
    'gin': {'input_dim': 7, 'num_layers': 10},
    'gat': {'input_dim': 7, 'num_layers': 8, 'num_heads': 8},  # More heads
    'transformer': {'input_dim': 7, 'num_layers': 6}
}
```

## Next Steps

1. ✓ Prepare IEEE grid datasets
2. ✓ Train models on one grid
3. ✓ Run batch evaluation
4. → Analyze cross-grid generalization
5. → Identify best model architecture
6. → Consider domain adaptation techniques for challenging pairs
