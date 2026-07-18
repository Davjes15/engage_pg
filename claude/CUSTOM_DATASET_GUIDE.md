# Using Custom Datasets with ENGAGE

## Directory Structure Required

Your custom dataset must follow this exact directory structure:

```
your-data-dir/
├── grid-name-1/
│   ├── train/
│   │   ├── dataset.pt          # PyTorch file with list of Data objects
│   │   └── dataset_src.csv     # CSV with source file paths
│   └── test/                   # (optional, not used currently)
│       └── dataset.pt
├── grid-name-2/
│   ├── train/
│   │   ├── dataset.pt
│   │   └── dataset_src.csv
│   └── test/
│       └── dataset.pt
└── grid-name-3/
    └── train/
        ├── dataset.pt
        └── dataset_src.csv
```

## Key Specifications

### 1. Grid Names Must Match Code
- Grid names must correspond to the codes used in experiment files
- For CC experiments: `get_dist_grid_codes(scenario=1)` returns valid grid names
- For custom grids: You define the names when creating the directory structure
- **Example valid names**: `"1-LV-rural1--0-no_sw"`, `"1-LV-urban--0-no_sw"`, `"my-grid-1"`, etc.

### 2. PyTorch Data Object Format (`dataset.pt`)

Each `dataset.pt` file must be a **list of PyTorch Geometric Data objects**.

Each Data object must have these exact attributes:

```python
Data(
    x=torch.Tensor,           # Shape: (N, 7)
    edge_index=torch.Tensor,  # Shape: (2, 2E), dtype=int64
    edge_attr=torch.Tensor,   # Shape: (2E, 4)
    y=torch.Tensor,           # Shape: (N, 4)
    dc_pf=torch.Tensor        # Shape: (N, 4)
)
```

### 3. Node Features (x) - Dimension (N, 7)
Where N = number of buses/nodes in the network

```python
x[:, 0]  # Slack bus indicator (1 if slack, 0 otherwise)
x[:, 1]  # PV generator indicator (1 if present, 0 otherwise)
x[:, 2]  # PQ load indicator (1 if present, 0 otherwise)
x[:, 3]  # p_mw (active power in MW) - may be NaN for slack buses
x[:, 4]  # q_mvar (reactive power in MVAr) - may be NaN for PV/slack buses
x[:, 5]  # vm_pu (voltage magnitude in per unit)
x[:, 6]  # va_degree (voltage angle in degrees)
```

**Important**: Some values should be NaN based on bus type:
- **Slack buses**: Set p_mw and q_mvar to NaN
- **PV buses**: Set q_mvar and va_degree to NaN
- **PQ buses**: Set vm_pu and va_degree to NaN

### 4. Ground Truth Labels (y) - Dimension (N, 4)
```python
y[:, 0]  # p_mw (actual active power)
y[:, 1]  # q_mvar (actual reactive power)
y[:, 2]  # vm_pu (actual voltage magnitude)
y[:, 3]  # va_degree (actual voltage angle)
```

### 5. Edge Features (edge_attr) - Dimension (2E, 4)
Where E = number of unique edges (the factor of 2 accounts for undirected edges)

```python
edge_attr[:, 0]  # trafo indicator (1 if transformer, 0 if line)
edge_attr[:, 1]  # r_pu (resistance in per-unit)
edge_attr[:, 2]  # x_pu (reactance in per-unit)
edge_attr[:, 3]  # sc_voltage (short-circuit voltage as %) - NaN for lines
```

### 6. DC Power Flow (dc_pf) - Dimension (N, 4)
```python
dc_pf[:, 0]  # p_mw (DC-PF approximated)
dc_pf[:, 1]  # q_mvar (DC-PF approximated, typically 0)
dc_pf[:, 2]  # vm_pu (DC-PF approximated)
dc_pf[:, 3]  # va_degree (DC-PF approximated)
```

This should be calculated using DC Power Flow analysis on your power grid, or you can use a similar physics-based approximation.

### 7. Edge Index (edge_index) - Dimension (2, 2E)

```python
edge_index[0, :]  # Source node indices
edge_index[1, :]  # Target node indices
```

**Important**: 
- Must represent an **undirected graph** (each edge appears twice, once in each direction)
- For each connection between buses i and j:
  - Include edge i→j
  - Include edge j→i
- Must use **dtype=torch.int64**
- Indices must be in range [0, N-1] (remapped to contiguous indices)

### 8. Dataset Source CSV (`dataset_src.csv`)

Simple CSV file with one column listing source files:

```csv
src
path/to/source/file1.json
path/to/source/file2.json
path/to/source/file3.json
```

- Can be any relative or absolute path (for reference only)
- Not strictly used by experiments, but required for consistency

## Data Type Requirements

```python
# Data type specifications:
x = torch.tensor(..., dtype=torch.float32)
y = torch.tensor(..., dtype=torch.float32)
edge_attr = torch.tensor(..., dtype=torch.float32)
edge_index = torch.tensor(..., dtype=torch.int64)  # IMPORTANT: int64, not int32
dc_pf = torch.tensor(..., dtype=torch.float32)
```

## Special Handling

### NaN Values
- NaNs are acceptable in x, edge_attr, and dc_pf
- The code uses `torch.nan_to_num()` to convert NaNs to 0s when needed
- Ensure NaNs appear where physically appropriate (based on bus type or edge type)

### Remapping
- Node indices must be contiguous (0 to N-1)
- If your original data has non-contiguous node IDs, remap them before creating edge_index

## How to Create Custom Dataset Programmatically

```python
import torch
from torch_geometric.data import Data

# For each grid, create a list of Data objects
dataset = []

for each_sample_in_your_grid:
    # Extract features from your power grid simulation/data
    x = torch.tensor(node_features, dtype=torch.float32)  # (N, 7)
    y = torch.tensor(true_values, dtype=torch.float32)    # (N, 4)
    edge_index = torch.tensor(edges, dtype=torch.int64)   # (2, 2E)
    edge_attr = torch.tensor(edge_features, dtype=torch.float32)  # (2E, 4)
    dc_pf = torch.tensor(dc_pf_values, dtype=torch.float32)  # (N, 4)
    
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        dc_pf=dc_pf
    )
    dataset.append(data)

# Save the dataset
torch.save(dataset, 'path/to/grid-name/train/dataset.pt')

# Create source CSV
import pandas as pd
pd.DataFrame(['source_file_1', 'source_file_2', ...], columns=['src']).to_csv(
    'path/to/grid-name/train/dataset_src.csv'
)
```

## Running Experiments with Custom Data

Once your dataset is prepared:

```bash
# Cross-context experiment
python cross_context_experiment.py \
  --data_dir your-data-dir/ \
  --model gcn \
  --epochs 100 \
  --save_results \
  --mmd

# Out-of-distribution experiment
python out_of_distribution_experiment.py \
  --data_dir your-data-dir/ \
  --model gcn \
  --epochs 100 \
  --save_results \
  --mmd
```

Note: The grid names passed to `training_grids` and `testing_grids` must match your directory names.

## Critical Validation Checklist

- [ ] Directory structure matches: `data_dir/grid-name/train/`
- [ ] Each `dataset.pt` is a list of Data objects (not a single Data object)
- [ ] All x tensors have shape (N, 7) with dtype float32
- [ ] All y tensors have shape (N, 4) with dtype float32
- [ ] All edge_index tensors have shape (2, 2E) with dtype int64
- [ ] All edge_attr tensors have shape (2E, 4) with dtype float32
- [ ] All dc_pf tensors have shape (N, 4) with dtype float32
- [ ] Edge indices are in range [0, N-1]
- [ ] Edges represent undirected graph (each edge appears twice)
- [ ] NaN values appear only where appropriate (masked features by bus type)
- [ ] `dataset_src.csv` exists with 'src' column

## Common Issues

### Issue: "Index out of bounds"
- **Cause**: Node IDs in edge_index are not contiguous (e.g., jump from 5 to 100)
- **Fix**: Remap all node IDs to [0, N-1] range

### Issue: "Shape mismatch" or "Wrong number of features"
- **Cause**: Feature matrices have wrong dimensions
- **Fix**: Verify x is (N, 7), y is (N, 4), edge_attr is (2E, 4)

### Issue: Model produces NaN predictions
- **Cause**: Input contains unexpected NaN values
- **Fix**: Check that only masked features are NaN, others are valid floats

### Issue: "AssertionError about directed/undirected"
- **Cause**: Graph is not properly undirected (missing reverse edges)
- **Fix**: For each edge i→j, ensure j→i also exists in edge_index

## Compatibility Notes

- Your data should represent **distribution grids** for power system analysis
- The models expect **electrical engineering** inputs (power, voltage, impedance)
- Different grid sizes (N varies) are okay and handled automatically
- Very small grids (N < 5) may cause issues with DC-PF calculations
- Very large grids (N > 1000) may require more GPU memory
