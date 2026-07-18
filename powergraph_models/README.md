# PowerGraph Models Setup

## Structure

```
/home/davjes/projects/engage/
├── powergraph_models/
│   ├── __init__.py          ← Package initialization with imports
│   ├── model.py             ← GNN model definitions from PowerGraph-Node
│   └── README.md            ← This file
└── eval_powergraph_models.py ← Evaluation script using these models
```

## What's Included

- **get_gnnNets()**: Factory function to create models by name
- **GCN**: Graph Convolutional Network
- **GIN**: Graph Isomorphism Network  
- **GAT**: Graph Attention Network
- **TRANSFORMER**: Graph Transformer Network

All models support:
- Graph-level regression
- Node-level regression (power flow, OPF)
- Configurable hidden dimensions and layers

## Usage

```python
from powergraph_models.model import get_gnnNets

model_params = {
    'model_name': 'gcn',  # or 'gin', 'gat', 'transformer'
    'hidden': 32,
    'num_layers': 2,
    'dropout': 0.5,
    'act': 'relu'
}

model = get_gnnNets(
    input_dim=4,          # Number of node features
    output_dim=4,         # Number of output dimensions
    model_params=model_params,
    graph_regression=False,
    node_pf_regression=True,
    node_opf_regression=False
)
```

## Integration with ENGAGE

The `eval_powergraph_models.py` script:
1. Loads PowerGraph-Node pre-trained models
2. Evaluates on new IEEE grids
3. Computes MMD (Maximum Mean Discrepancy)
4. Calculates NRMSE (Normalized Root Mean Square Error)
5. Generates ENGAGE g-score

### Example Command

```bash
python eval_powergraph_models.py \
    --train_data ieee-grids/IEEE24/processed_node/data.pt \
    --test_data ieee-grids/IEEE39/processed_node/data.pt \
    --model_path ./models/ieee24_gcn.pth \
    --model_type gcn \
    --output_file results.csv
```

## Notes

- This is a copy from PowerGraph-Node repository
- All PyTorch Geometric dependencies already installed
- Models trained on PowerGraph data format (batched, flattened nodes)
- Compatible with ENGAGE's g-score metric calculations
