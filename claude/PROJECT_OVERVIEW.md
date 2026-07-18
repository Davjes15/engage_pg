# ENGAGE Project Overview

## Project Purpose

**ENGAGE: Evaluating Network Generalization for AC Grid Estimation**

This project evaluates how Graph Neural Networks (GNNs) generalize across different electrical distribution grids. It measures model performance on unseen grids and correlates this with graph structural differences (MMD - Maximum Mean Discrepancy) to derive a "generalization score."

---

## Key Components

### 1. Data Structure

Each sample in the dataset represents a power grid as a graph with:

- **Nodes (x)**: Dimension `(N, 7)`
  - N = number of buses/nodes in the network
  - Features: `[Slack?, PV?, PQ?, p_mw, q_mvar, vm_pu, va_degree]`
  
- **Edges (edge_index)**: Dimension `(2, 2E)`
  - E = number of lines
  - Represents directed edges
  
- **Edge Attributes (edge_attr)**: Dimension `(2E, 4)`
  - Features: `[trafo?, r_pu, x_pu, sc_voltage]`
  
- **Ground Truth (y)**: Dimension `(N, 4)`
  - True values: `[p_mw, q_mvar, vm_pu, va_degree]`
  
- **DC Power Flow Approximation (dc_pf)**: Dimension `(N, 4)`
  - DC-PF approximated values for same features as y

### 2. Data Source

- Data comes from **SimBench** - simulated electrical distribution grids
- Supports multiple scenarios (LV, MV, combinations)
- Different "no_sw" (no switch) configurations

---

## Experimental Framework

### Two Main Experiments

#### A. Cross-Context (CC) Experiment
**File**: [cross_context_experiment.py](cross_context_experiment.py)

**Purpose**: Test how models trained on one distribution grid perform on other grids

**Test Cases**: All **permutations** of grid pairs (A→B, B→A, etc.)

```
For N grids: N * (N-1) ordered pairs
```

#### B. Out-of-Distribution (OOD) Experiment
**File**: [out_of_distribution_experiment.py](out_of_distribution_experiment.py)

**Purpose**: Test models trained on all-but-one grid, tested on the held-out grid

**Test Cases**: One grid left out, train on remaining N-1 grids, test on held-out grid

```
For N grids: N combinations
```

---

## Experiment Execution Pipeline

### Step 1: Data Generation
**File**: [graph_gen.py](graph_gen.py)

```bash
python graph_gen.py --size 5 --scenario 1
```

Generates training/validation/test splits for each grid type.

### Step 2: Run Cross-Context or OOD Experiments

#### Command-line Example:
```bash
python cross_context_experiment.py \
  --data_dir outputs/2025-01-15_10:58:53/ \
  --model gcn \
  --epochs 100 \
  --save_results \
  --mmd \
  --dc_pf
```

#### Arguments:
- `--data_dir`: Base directory with generated datasets
- `--model`: Model type ('gcn' or 'arma_gnn')
- `--epochs`: Training epochs
- `--save_results`: Save CSV results
- `--mmd`: Calculate MMD (graph metric)
- `--dc_pf`: Include DC power flow baseline
- `--plot`: Generate loss curves
- `--skip_experiment`: Skip performance evaluation

### Step 3: Results Processing

#### For each test case, the experiment produces:

1. **Performance Metrics** (nrmse_test.csv):
   - Training grid, testing grid
   - Feature variations (cycles, path_lengths, degree)
   - NRMSE (Normalized Root Mean Square Error)
   - Best validation loss
   - Training time, total epochs

2. **MMD Metrics** (mmd.csv):
   - Degree distribution MMD
   - Laplacian spectrum MMD

3. **Generalization Statistics** (gen_stats.csv):
   - Mean NRMSE
   - Std deviation
   - MMD ranges
   - **G-score (Generalization Score)**

---

## Model Variations

The experiments test models with different feature augmentations:

| Variation | Cycles | Path Lengths | Degree |
|-----------|--------|--------------|--------|
| ref       | ✗      | ✗            | ✗      |
| cycles    | ✓      | ✗            | ✗      |
| paths     | ✗      | ✓            | ✗      |
| degree    | ✗      | ✗            | ✓      |
| cycles+paths | ✓   | ✓            | ✗      |
| all       | ✓      | ✓            | ✓      |

**Augmented Features**:
- **Cycles**: Shortest cycle length containing each node
- **Path Lengths**: Shortest path from each node to slack bus
- **Degree**: Node degree in the graph

---

## Models

### Available Models
**File**: [models.py](models.py)

1. **GCN** (Graph Convolutional Network)
   - Standard baseline

2. **ARMA_GNN** (Autoregressive Moving Average GNN)
   - Based on: Hansen et al., "Power Flow Balancing With Decentralized Graph Neural Networks"
   - 8 ARMA layers with 5 stacks
   - LeakyReLU activation
   - Input: 7-dimensional node features
   - Output: 4-dimensional predictions

---

## How G-Score Is Calculated

**Location**: [training_utils.py](training_utils.py#L316), function `get_generalization_score()`

### Formula and Algorithm

```python
def get_generalization_score(mmds, nrmses, alpha=1.0):
    # Step 1: Calculate 2nd and 98th percentiles of NRMSE
    p_min = np.percentile(nrmses, 2)
    p_max = np.percentile(nrmses, 98)
    
    # Step 2: Mask to only include points in the 2-98 percentile range
    p_where = (nrmses <= p_max) & (nrmses >= p_min)
    
    # Step 3: Calculate MMD range within the percentile range
    min_mmd_percentile = mmds.min(where=p_where, initial=mmds.max())
    max_mmd_percentile = mmds.max(where=p_where, initial=mmds.min())
    mmd_range_percentile = max_mmd_percentile - min_mmd_percentile
    
    # Step 4: Calculate mean and std of NRMSE within percentile range
    mean_nrmse_percentile = nrmses.mean(where=p_where)
    std_rmse_percentile = nrmses.std(where=p_where)
    
    # Step 5: Calculate generalization score
    eps = 1e-8
    score = mean_nrmse_percentile + alpha * std_rmse_percentile * (
        np.log(mmd_range_percentile + 1) / (mmd_range_percentile + eps)
    )
    
    return mean_nrmse_percentile, std_rmse_percentile, mmd_range_percentile, score
```

### G-Score Components

The returned tuple contains:

1. **mean_nrmse_percentile**: Average NRMSE for central 96% of results
2. **std_rmse_percentile**: Std deviation of NRMSE for central 96%
3. **mmd_range_percentile**: Range of MMD values (max - min) for central 96%
4. **g_score**: The generalization score

### G-Score Interpretation

The g-score formula is:

$$g = \bar{\text{NRMSE}} + \alpha \cdot \sigma_{\text{NRMSE}} \cdot \frac{\ln(\text{MMD\_range} + 1)}{\text{MMD\_range} + \epsilon}$$

Where:
- $\bar{\text{NRMSE}}$: Mean prediction error (lower is better)
- $\sigma_{\text{NRMSE}}$: Variability in performance
- $\text{MMD\_range}$: Diversity of training vs test graphs
- $\alpha = 1.0$: Weighting parameter (default)
- $\epsilon = 1e-8$: Numerical stability

**Interpretation**:
- **Lower g-score = Better generalization**
- The score combines prediction error with a penalty term that accounts for MMD range
- When MMD range is small, the penalty term is small (graphs are similar)
- When MMD range is large, the log term grows slowly (sublinear with MMD_range)
- The percentile-based filtering (2-98%) removes outliers that could skew the metric

---

## MMD (Maximum Mean Discrepancy) Metrics

**Purpose**: Measure structural similarity between training and test graph distributions

**Calculated in**: [training_utils.py](training_utils.py#L280), function `evaluate_mmd()`

### Two Metrics:

1. **Degree Distribution MMD** (`sigma=1e2`)
   - Compares degree distributions between graph sets
   - Higher sigma = smoother kernel

2. **Laplacian Spectrum MMD** (`sigma=1e-2`)
   - Compares normalized Laplacian eigenvalue spectra
   - Captures global graph structure

### Calculation Process:
1. Convert PyTorch Geometric graphs to NetworkX
2. Extract descriptor function (degree or Laplacian spectrum)
3. Apply Gaussian kernel
4. Compute MMD using GGME library
5. Return scalar MMD distance for each metric

---

## Training Process

**Key Details**:

- **Loss Function**: Weighted MSE (inverse norm weighting)
- **Optimizer**: Adam (learning_rate=1e-3)
- **Early Stopping**: Yes (patience=500 epochs)
- **Best Weights**: Restored from best validation checkpoint
- **Metric**: NRMSE normalized by feature range

### Weighted MSE Loss:
```python
# Weight each sample by 1/||y_true||_2
# Gives equal importance to small and large predictions
```

### NRMSE (Test Metric):
```python
NRMSE = RMSE / avg_range
# Where avg_range = average feature range across all dimensions
```

---

## File Structure

```
engage/
├── cross_context_experiment.py    # Main CC experiment
├── out_of_distribution_experiment.py # Main OOD experiment
├── example.py                      # Simple example
├── models.py                       # GNN model definitions
├── training_utils.py              # Training, evaluation, g-score logic
├── graph_utils.py                 # Graph processing utilities
├── graph_gen.py                   # Data generation
├── base_gen_config.yaml           # Data generation config
├── ggme/                          # Submodule: Graph ML evaluation
└── powerdata-gen/                 # Submodule: Power grid data generation
```

---

## Results Output

### CSV Files Generated:

1. **results_cc.csv** (Cross-Context)
   - Rows: Each train-test grid pair + feature variation combo
   - Columns: Grid names, feature flags, NRMSE, loss, time, epochs

2. **results_cc_mmd.csv**
   - Rows: Grid pairs
   - Columns: MMD_degree, MMD_laplacian

3. **results_cc_gen_stats.csv**
   - Rows: Model variations (ref, cycles, paths, degree, cycles+paths, all, dc_pf)
   - Columns: mean_nrmse, std_nrmse, mmd_range_degree, mmd_range_laplacian, g_score_degree, g_score_laplacian

---

## Baseline: DC Power Flow (DC-PF)

- Provides a reference performance level
- Calculated via: [training_utils.py](training_utils.py#L271), function `test_dc_pf()`
- Uses physics-based DC power flow approximation instead of learned model
- Included in generalization statistics for comparison

---

## Summary

ENGAGE systematically evaluates GNN generalization across power grids by:

1. **Generating synthetic power grid datasets** from SimBench
2. **Training models** on specific grids with feature augmentations
3. **Testing on held-out grids** (both CC and OOD scenarios)
4. **Measuring graph dissimilarity** via MMD metrics
5. **Computing g-score** that combines prediction error + MMD diversity
6. **Comparing model variants** to understand which features aid generalization
