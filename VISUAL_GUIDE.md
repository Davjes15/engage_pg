# Visual Guide: IEEE Transmission Grid Model Evaluation

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    YOUR TRANSMISSION GRIDS                      │
│  IEEE18  │  IEEE24  │  IEEE39  │  UK (proprietary)            │
└──────────┼──────────┼──────────┼──────────────────────────────┘
           │          │          │
           ▼          ▼          ▼
┌─────────────────────────────────────────────────────────────────┐
│              PyTorch Geometric Format                            │
│  [Data(x, edge_index, edge_attr, y, dc_pf), ...]              │
│  Saved: ieee-grids/{grid}/train/dataset.pt                     │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│              Train Models on ONE Grid                            │
│           (e.g., IEEE18 as training grid)                       │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ GCN (Fast)  ──→  ieee18_gcn.pth                        │   │
│  │ GIN (Medium)──→  ieee18_gin.pth                        │   │
│  │ GAT (Medium)──→  ieee18_gat.pth                        │   │
│  │ Transformer───→  ieee18_transformer.pth               │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│            Evaluate on THREE Test Grids                         │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ eval_ieee_models.py / batch_eval_ieee.py             │    │
│  │                                                        │    │
│  │ Input:   trained_models/ieee18_*.pth                 │    │
│  │ + Test:  ieee-grids/{IEEE24,IEEE39,UK}/...          │    │
│  └────────────────────────────────────────────────────────┘    │
│                        │                                         │
│        ┌───────────────┼───────────────┐                        │
│        ▼               ▼               ▼                        │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐                   │
│  │ IEEE24   │   │ IEEE39   │   │   UK     │                   │
│  └──────────┘   └──────────┘   └──────────┘                   │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│              For Each Evaluation:                               │
│                                                                  │
│  1. NRMSE_test    ─→ Prediction error on test grid            │
│  2. MMD_degree    ─→ Graph structure similarity (node degree)  │
│  3. MMD_laplacian ─→ Graph structure similarity (spectrum)     │
│  4. G-SCORE       ─→ Generalization metric (combined)          │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     RESULTS MATRIX                              │
│                                                                  │
│               IEEE24    IEEE39      UK                          │
│  ┌──────────┬──────────┬──────────┬──────────┐                │
│  │ GCN      │ 0.142    │ 0.165    │ 0.234    │                │
│  │ GIN      │ 0.134    │ 0.148    │ 0.215    │                │
│  │ GAT      │ 0.128    │ 0.142    │ 0.198    │  ← G-SCORES   │
│  │ Transform│ 0.124    │ 0.138    │ 0.185    │                │
│  └──────────┴──────────┴──────────┴──────────┘                │
│                                                                  │
│  ✓ Lower is better                                            │
│  ✓ Transformer shows best generalization                      │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│            Insights & Decisions                                 │
│                                                                  │
│  • Which model generalizes best?  → Transformer               │
│  • How does grid size affect it?  → Larger grids = higher     │
│  • Is domain shift significant?   → MMD shows yes             │
│  • Should we use augmented features? → Test with flags        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Format

### Input: dataset.pt
```
List[
    Data(
        x: (N, 7)           # Node features
        ├─ [0] Slack?
        ├─ [1] PV?
        ├─ [2] PQ?
        ├─ [3] p_mw
        ├─ [4] q_mvar
        ├─ [5] vm_pu
        └─ [6] va_degree
        
        edge_index: (2, 2E)  # Directed edges (undirected doubled)
        
        edge_attr: (2E, 4)   # Edge features
        ├─ [0] trafo?
        ├─ [1] r_pu
        ├─ [2] x_pu
        └─ [3] sc_voltage
        
        y: (N, 4)            # Ground truth
        ├─ [0] p_mw
        ├─ [1] q_mvar
        ├─ [2] vm_pu
        └─ [3] va_degree
        
        dc_pf: (N, 4)        # DC power flow
        └─ Same as y
    ),
    ...
]
```

---

## Workflow Diagram

```
PREPARE DATA                    TRAIN MODELS                 EVALUATE
═════════════════════════════════════════════════════════════════════

IEEE18 Grid ─┐
IEEE24 Grid ─┼→ Convert to   ──→ GCN model  ────→ Train on IEEE18
IEEE39 Grid ─┤   PyTorch         GIN model         (Get .pth files)
UK Grid ────┘    Format          GAT model  
                 (dataset.pt)    Transformer ──→ Evaluate on:
                                                  • IEEE24
                                                  • IEEE39
                                                  • UK
                                                  
                                                ──→ Calculate:
                                                    • NRMSE
                                                    • MMD
                                                    • G-Score
                                                    
                                                ──→ Results CSV
```

---

## Model Selection Guide

```
                        COMPLEXITY →
        
       GCN         GIN         GAT      TRANSFORMER
       ───────────────────────────────────────────
                    
SPEED  ████████░░  ██████░░░░  ██████░░░░  ████░░░░░░  ← ACCURACY
       Fast        Medium      Medium      Slow
       
PARAMS ~50K        ~50K        ~60K        ~70K
       
BEST   Baseline    Complex     Adaptive    High-Accuracy
FOR    (TX)        Topology    Features    (Research)
       
USE    ✓ Always    ✓ If        ✓ Often    ✓ If time
WHEN   try first   complex     best for   permits
                   needed      TX grids
```

---

## Script Usage Flow

```
SINGLE EVALUATION
═════════════════════════════════════════════════════════════════════

$ python eval_ieee_models.py \
    --data_dir ./ieee-grids \
    --training_grid IEEE18 \
    --test_grid IEEE24 \
    --model_path ./trained_models/ieee18_gcn.pth \
    --model_type gcn
    
    ↓
    
Output: evaluation_results.csv
┌────────────────────────────────────────────────────────────┐
│ training_grid,test_grid,nrmse_test,g_score_degree,...    │
│ IEEE18,IEEE24,0.125,0.142,...                             │
└────────────────────────────────────────────────────────────┘


BATCH EVALUATION
═════════════════════════════════════════════════════════════════════

$ python batch_eval_ieee.py \
    --data_dir ./ieee-grids \
    --model_dir ./trained_models \
    --training_grid IEEE18 \
    --models gcn gin gat transformer
    
    ↓
    [Progress: 1/12] IEEE18→IEEE24 (GCN)    ✓
    [Progress: 2/12] IEEE18→IEEE24 (GIN)    ✓
    [Progress: 3/12] IEEE18→IEEE24 (GAT)    ✓
    [Progress: 4/12] IEEE18→IEEE24 (Trans)  ✓
    [Progress: 5/12] IEEE18→IEEE39 (GCN)    ✓
    ...
    
    ↓
    
Output: 
├─ batch_results_IEEE18.csv      (12 rows: 4 models × 3 test grids)
└─ summary_IEEE18.csv             (4 rows: summary per model)
```

---

## G-Score Interpretation

```
G-Score Scale:
═════════════════════════════════════════════════════════════════════

  0.05    0.10    0.15    0.20    0.25    0.30    0.50
  |-------|-------|-------|-------|-------|-------|-----|
  ✓✓✓     ✓✓      ✓       ◐       ◑       ◑       ✗
  
  Excellent    Good   Acceptable   Fair    Poor   Challenging
  Same Grid   Similar  Different  Larger Very     Transmission
  or very     Size    Size       Diff    Large    Grid Shift
  close
  
Expected Ranges (trained on IEEE18):
─────────────────────────────────────
→ IEEE24:  0.10-0.20  (similar size)
→ IEEE39:  0.15-0.25  (larger)
→ UK:      0.25-0.50+ (very different)

Higher = harder domain shift
```

---

## File Organization

```
YOUR PROJECT
│
├── ieee-grids/
│   ├── IEEE18/
│   │   └── train/
│   │       ├── dataset.pt          ← PyTorch data
│   │       └── dataset_src.csv     ← Metadata
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
│
├── trained_models/               ← Save here
│   ├── ieee18_gcn.pth
│   ├── ieee18_gin.pth
│   ├── ieee18_gat.pth
│   └── ieee18_transformer.pth
│
├── results/                       ← Generated here
│   ├── batch_results_IEEE18.csv
│   └── summary_IEEE18.csv
│
└── ENGAGE repo files
    ├── models.py                  ← (Updated)
    ├── eval_ieee_models.py        ← (New)
    ├── batch_eval_ieee.py         ← (New)
    └── IEEE_*.md                  ← (New docs)
```

---

## Performance Metrics Explained

```
NRMSE (Prediction Accuracy)
───────────────────────────────────────────────────────────────
Formula: √(MSE) / feature_range_average
Range: Typically 0.01 - 0.3
Lower = More Accurate
Example: 0.125 means average 12.5% error

MMD_degree (Node Degree Similarity)
───────────────────────────────────────────────────────────────
Measures: How similar the node degree distributions are
Higher = More Different
Example: IEEE18(18 nodes) vs UK(1000s nodes) → High MMD

MMD_laplacian (Graph Spectrum Similarity)
───────────────────────────────────────────────────────────────
Measures: How similar the graph eigenvalue spectra are
Higher = More Different
Example: Dense transmission grid vs sparse distribution → High MMD

G-SCORE (Generalization Metric)
───────────────────────────────────────────────────────────────
Formula: mean_nrmse + std_nrmse × log(mmd_range+1)/(mmd_range+ε)
Combines: Accuracy + Reliability - Domain Shift
Lower = Better Generalization
Example: 0.142 combines NRMSE=0.125 with MMD adjustment
```

---

## Common Issues & Visual Solutions

```
ISSUE: High G-Score (>0.4)
───────────────────────────────────────────────────────────────
Likely causes:

1. Domain Shift Too Large
   ┌──────────┐              ┌──────────┐
   │ IEEE18   │              │    UK    │
   │ (small)  │─── MMD ───→ │(very big)│
   │ 18 nodes │    HIGH     │1000+ nodes
   └──────────┘              └──────────┘
   
   Solution: This is expected! Use domain adaptation techniques
   
2. Model Undertrained
   ┌──────────────────┐
   │ Training Loss \\  │ ← Still dropping
   │              \\   │
   └──────────────────┘
   
   Solution: Train longer, use better hyperparameters
   
3. Test Data Quality
   ┌─────────────────┐
   │ NaN in features │ ← Check data format
   │ Wrong shape     │
   └─────────────────┘
   
   Solution: Verify dataset.pt has correct dimensions


ISSUE: Transformer Much Slower Than GCN
───────────────────────────────────────────────────────────────

Speed Comparison (1000-sample grid):
┌─────────────┬────────┐
│ GCN:        │ ███    │ 1-2s
│ GIN:        │ ████   │ 1.5-2.5s
│ GAT:        │ ██████ │ 2-3.5s
│ Transformer:│ ██████████████ │ 4-8s
└─────────────┴────────┘

Solution: Use Transformer only if:
• You have GPU memory
• Accuracy improvement justifies time
• Batch evaluation (parallel runs)


ISSUE: Grid Not Found
───────────────────────────────────────────────────────────────

Check structure:
✓ ieee-grids/
  ✓ IEEE18/
    ✓ train/
      ✓ dataset.pt      ← Missing?
      ✗ dataset_src.csv ← Not critical

Solution: Use lowercase and exact naming
```

---

## Command Reference Card

```
BASIC COMMANDS
═════════════════════════════════════════════════════════════════

Single eval:
  python eval_ieee_models.py --data_dir ./ieee-grids \
    --training_grid IEEE18 --test_grid IEEE24 \
    --model_path ./models/ieee18_gcn.pth --model_type gcn

Batch eval:
  python batch_eval_ieee.py --data_dir ./ieee-grids \
    --model_dir ./trained_models --training_grid IEEE18 \
    --models gcn gin gat transformer

With features:
  python eval_ieee_models.py ... --add_cycles --add_path_lengths

Different batch size:
  python eval_ieee_models.py ... --batch_size 32

Verbose output:
  python eval_ieee_models.py ... --verbose

Help:
  python eval_ieee_models.py --help
  python batch_eval_ieee.py --help
```

---

**Ready to start?** See `IEEE_QUICK_START.md` for next steps!
