#!/usr/bin/env python
"""
Evaluate PowerGraph-Node Pre-trained Models with ENGAGE g-score Metrics
=========================================================================

Load a model trained with PowerGraph-Node on one grid, test on MULTIPLE other grids,
and compute ENGAGE's g-score using MMD + NRMSE across all test contexts.

The g-score measures generalization by capturing how NRMSE varies with topological
distance (MMD) across different test grids. Lower g-score = better generalization.

Usage:
    # Evaluate IEEE118 model on IEEE24, IEEE39, and UK grids
    python eval_powergraph_models.py \
        --train_data ./ieee-grids/IEEE118/processed_node/data.pt \
        --test_data ./ieee-grids/IEEE24/processed_node/data.pt \
                    ./ieee-grids/IEEE39/processed_node/data.pt \
                    ./ieee-grids/UK/processed_node/data.pt \
        --model_path ./trained_models/IEEE118/gcn_3l_32h_best.pth \
        --model_type gcn

    # Or use a directory to test on all grids
    python eval_powergraph_models.py \
        --train_data ./ieee-grids/IEEE118/processed_node/data.pt \
        --test_data_dir ./ieee-grids \
        --model_path ./trained_models/IEEE118/gcn_3l_32h_best.pth \
        --model_type gcn
"""

import argparse
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import warnings

warnings.filterwarnings('ignore')

# Add ENGAGE's ggme module to path for MMD functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ggme', 'src'))
from metrics.mmd import mmd as engage_mmd
from metrics.kernels import gaussian_kernel

# Import NRMSE function from ENGAGE
from training_utils import nrmse_range


def get_generalization_score_raw(mmds, nrmses, alpha=1.0):
    """
    Compute g-score WITHOUT percentile trimming.
    
    This is equivalent to ENGAGE's get_generalization_score but uses all samples
    instead of filtering to 2nd-98th percentile. This is necessary when you have
    few test contexts (< 10 samples).
    
    Formula: g_score = mean_nrmse + alpha * std_nrmse * log(mmd_range + 1) / (mmd_range + eps)
    
    Lower g-score = better generalization.
    """
    eps = 1e-8
    
    mean_nrmse = nrmses.mean()
    std_nrmse = nrmses.std()
    mmd_range = mmds.max() - mmds.min()
    
    # The penalty term: high std with low mmd_range is bad
    # log(mmd_range + 1) / (mmd_range + eps) approaches 1 as mmd_range -> 0
    score = mean_nrmse + alpha * std_nrmse * (np.log(mmd_range + 1) / (mmd_range + eps))
    
    return mean_nrmse, std_nrmse, mmd_range, score


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate PowerGraph-Node models with ENGAGE g-score metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--train_data',
        required=True,
        help='Path to training data.pt (the grid the model was trained on)'
    )
    parser.add_argument(
        '--test_data',
        nargs='+',
        help='Paths to one or more test data.pt files (evaluation grids)'
    )
    parser.add_argument(
        '--test_data_dir',
        help='Directory containing grid folders (will test on all grids except train)'
    )
    parser.add_argument(
        '--model_path',
        required=True,
        help='Path to pre-trained model weights (.pth file)'
    )
    parser.add_argument(
        '--model_type',
        default='gcn',
        choices=['gcn', 'gin', 'gat', 'transformer'],
        help='Type of model architecture'
    )
    parser.add_argument(
        '--output_file',
        default='powergraph_evaluation_results.csv',
        help='Output CSV file for results'
    )
    parser.add_argument(
        '--cached_mmds',
        help='Path to precomputed pairwise_mmds.csv (from precompute_grid_mmds.py)'
    )
    parser.add_argument(
        '--sigma_degree',
        type=float,
        default=1e1,
        help='Sigma parameter for degree distribution MMD (ignored if --cached_mmds provided)'
    )
    parser.add_argument(
        '--sigma_laplacian',
        type=float,
        default=1e-2,
        help='Sigma parameter for Laplacian spectrum MMD (ignored if --cached_mmds provided)'
    )
    
    args = parser.parse_args()
    
    # Validate: need either test_data or test_data_dir
    if not args.test_data and not args.test_data_dir:
        parser.error("Either --test_data or --test_data_dir is required")
    
    return args


def load_data(data_path):
    """Load PowerGraph-Node batched data"""
    print(f"  Loading: {data_path}")
    data_tuple = torch.load(data_path, weights_only=False)
    
    if isinstance(data_tuple, tuple) and len(data_tuple) == 2:
        graph_data, metadata = data_tuple
    else:
        graph_data = data_tuple
        metadata = None
    
    return graph_data, metadata


def get_grid_name(data_path):
    """Extract grid name from data path"""
    # e.g., ./ieee-grids/IEEE118/processed_node/data.pt -> IEEE118
    path = Path(data_path)
    if path.parent.name == 'processed_node':
        return path.parent.parent.name
    return path.parent.name


def discover_test_grids(test_data_dir, train_data_path):
    """Find all grids in directory except the training grid"""
    train_grid = get_grid_name(train_data_path)
    test_paths = []
    
    for grid_dir in Path(test_data_dir).iterdir():
        if grid_dir.is_dir() and grid_dir.name != train_grid:
            data_file = grid_dir / 'processed_node' / 'data.pt'
            if data_file.exists():
                test_paths.append(str(data_file))
    
    return sorted(test_paths)


def parse_model_filename(model_path):
    """Extract hyperparameters from model filename"""
    filename = os.path.basename(model_path)
    
    num_layers = 3  # default
    hidden_dim = 32  # default
    
    import re
    
    # Pattern: gcn_3l_32h_best.pth
    layer_match = re.search(r'(\d+)l', filename)
    if layer_match:
        num_layers = int(layer_match.group(1))
    
    hidden_match = re.search(r'(\d+)h', filename)
    if hidden_match:
        hidden_dim = int(hidden_match.group(1))
    
    return num_layers, hidden_dim


def load_model(model_type, model_path, input_dim, output_dim, edge_dim, device):
    """Load a PowerGraph-Node model"""
    # Import PowerGraph model classes
    from powergraph_models.model import GCN, GIN, GAT, TRANSFORMER
    
    model_classes = {
        'gcn': GCN,
        'gin': GIN,
        'gat': GAT,
        'transformer': TRANSFORMER
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}")
    
    num_layers, hidden_dim = parse_model_filename(model_path)
    print(f"  Model hyperparams from filename: {num_layers} layers, {hidden_dim} hidden dim")
    
    # PowerGraph models expect model_params dict, not individual kwargs
    model_params = {
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'edge_dim': edge_dim,
        'heads': 4,
        'readout': 'identity',  # Node-level prediction (no graph pooling)
        'dropout': 0.0
    }
    
    # Create model with expected signature
    model = model_classes[model_type](
        input_dim=input_dim,
        output_dim=output_dim,
        model_params=model_params,
        graph_regression=False,
        node_pf_regression=True,  # Power flow node regression
        node_opf_regression=False
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict):
        if 'net' in checkpoint:
            state_dict = checkpoint['net']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"  ✓ Model loaded: {model_type.upper()} ({sum(p.numel() for p in model.parameters()):,} params)")
    
    return model


def get_nodes_per_graph(data, metadata):
    """Extract nodes per graph from metadata"""
    if metadata is not None and 'x' in metadata:
        x_idx = metadata['x']
        if len(x_idx) > 1:
            return (x_idx[1] - x_idx[0]).item()
    return None


def compute_degree_distribution(data, nodes_per_graph=None):
    """
    Compute degree distribution for PowerGraph-Node data.
    Returns shape (1, num_nodes) since all samples share the same topology.
    
    Optimized: Uses vectorized operations and only processes edges from
    the first sample (since topology is shared across all samples).
    """
    edge_index = data.edge_index
    num_nodes = data.x.shape[0]
    
    if edge_index.numel() == 0:
        if nodes_per_graph:
            return np.zeros((1, nodes_per_graph))
        return np.zeros((1, num_nodes))
    
    max_edge_idx = edge_index.max().item()
    
    if nodes_per_graph is not None:
        n = nodes_per_graph
    elif max_edge_idx < num_nodes // 10:
        n = max_edge_idx + 1
    else:
        n = num_nodes
    
    # Get local edge indices
    edge_idx_local = edge_index[0] % n if nodes_per_graph else edge_index[0]
    
    # Vectorized degree counting using bincount (ensure CPU for numpy conversion)
    edge_idx_clipped = edge_idx_local[edge_idx_local < n].cpu()
    degrees = torch.bincount(edge_idx_clipped, minlength=n).float()
    
    # Since edges are repeated for each sample, divide by number of samples
    if nodes_per_graph:
        num_samples = num_nodes // nodes_per_graph
        degrees = degrees / num_samples
    
    return degrees.numpy().reshape(1, -1)


def compute_laplacian_spectrum(data, k=10, nodes_per_graph=None):
    """
    Compute k smallest non-trivial Laplacian eigenvalues.
    Returns shape (1, k) since all samples share the same topology.
    
    Optimized: Uses vectorized operations and unique edges.
    """
    edge_index = data.edge_index
    num_nodes = data.x.shape[0]
    
    if edge_index.numel() == 0:
        return np.zeros((1, k))
    
    max_edge_idx = edge_index.max().item()
    
    if nodes_per_graph is not None:
        n = nodes_per_graph
    elif max_edge_idx < num_nodes // 10:
        n = max_edge_idx + 1
    else:
        return np.zeros((1, k))  # Too large, skip
    
    # Get local edge indices (move to CPU for processing)
    src = (edge_index[0] % n if nodes_per_graph else edge_index[0]).cpu()
    dst = (edge_index[1] % n if nodes_per_graph else edge_index[1]).cpu()
    
    # Filter to valid edges and get unique (topology is same for all samples)
    valid_mask = (src < n) & (dst < n)
    src = src[valid_mask]
    dst = dst[valid_mask]
    
    # Stack and get unique edges
    edges = torch.stack([src, dst], dim=0)
    unique_edges = torch.unique(edges, dim=1)
    
    # Build adjacency matrix using vectorized indexing
    adj = torch.zeros(n, n)
    adj[unique_edges[0], unique_edges[1]] = 1.0
    
    # Compute Laplacian
    degrees = adj.sum(dim=1)
    L = torch.diag(degrees) - adj
    
    try:
        eigenvalues = torch.linalg.eigvalsh(L)
        eigs = eigenvalues[1:k+1].numpy()
        if len(eigs) < k:
            eigs = np.pad(eigs, (0, k - len(eigs)))
        return eigs.reshape(1, -1)
    except:
        return np.zeros((1, k))


def compute_mmd(X, Y, sigma):
    """
    Compute Maximum Mean Discrepancy using ENGAGE's unbiased estimator.
    """
    from functools import partial
    from metrics.kernels import KernelDistributionWrapper
    
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
    
    X_list = [x for x in X]
    Y_list = [y for y in Y]
    
    kernel = partial(gaussian_kernel, sigma=1.0/(2*sigma**2))
    kernel_wrapper = KernelDistributionWrapper(kernel, pad=True)
    
    mmd_squared = engage_mmd(X_list, Y_list, kernel=kernel_wrapper)
    
    return np.sqrt(max(mmd_squared, 0))


def compute_nrmse(predictions, targets, mask=None):
    """
    Compute NRMSE using ENGAGE's normalization (by range).
    
    NRMSE = RMSE / mean(range_per_dimension)
    """
    if mask is not None:
        predictions = predictions[mask]
        targets = targets[mask]
    
    # Convert to torch for consistency with ENGAGE
    pred_t = torch.from_numpy(predictions).float()
    targ_t = torch.from_numpy(targets).float()
    
    # Use ENGAGE's nrmse_range function
    nrmse = nrmse_range(targ_t, pred_t).item()
    
    # Also compute RMSE for reference
    rmse = np.sqrt(np.mean((predictions - targets)**2))
    
    return nrmse, rmse


def run_inference(model, data, device):
    """Run model inference on data"""
    with torch.no_grad():
        data = data.to(device)
        
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            predictions = model(data.x, data.edge_index, data.edge_attr)
        else:
            predictions = model(data.x, data.edge_index)
        
        targets = data.y
        mask = data.mask.cpu().numpy() if hasattr(data, 'mask') else None
        
        return predictions.cpu().numpy(), targets.cpu().numpy(), mask


def evaluate(args):
    """Main evaluation function - cross-context evaluation"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*80}")
    print("POWERGRAPH MODEL CROSS-CONTEXT EVALUATION")
    print(f"{'='*80}")
    print(f"Device: {device}")
    
    # ========================================================================
    # Load cached MMDs if provided
    # ========================================================================
    cached_mmds = None
    if args.cached_mmds:
        print(f"\nLoading cached MMDs from: {args.cached_mmds}")
        cached_mmds = pd.read_csv(args.cached_mmds)
        print(f"  ✓ Loaded {len(cached_mmds)} pairwise MMD values")
    
    # ========================================================================
    # Discover test grids
    # ========================================================================
    if args.test_data_dir:
        test_paths = discover_test_grids(args.test_data_dir, args.train_data)
        print(f"\nDiscovered {len(test_paths)} test grids in {args.test_data_dir}")
    else:
        test_paths = args.test_data
    
    train_grid = get_grid_name(args.train_data)
    test_grids = [get_grid_name(p) for p in test_paths]
    
    print(f"\nTrain grid: {train_grid}")
    print(f"Test grids: {test_grids}")
    print(f"Model: {args.model_path}")
    print(f"Model type: {args.model_type}")
    
    # ========================================================================
    # Load training data
    # ========================================================================
    print(f"\n{'-'*80}")
    print("Loading training data...")
    print(f"{'-'*80}")
    
    train_data, train_meta = load_data(args.train_data)
    train_nodes = get_nodes_per_graph(train_data, train_meta)
    print(f"  ✓ Train grid: {train_grid} ({train_nodes} nodes)")
    
    # Only compute topological features if not using cached MMDs
    train_degrees = None
    train_laplacian = None
    if cached_mmds is None:
        print("  Computing topological features (use --cached_mmds to skip)...")
        train_degrees = compute_degree_distribution(train_data, train_nodes)
        train_laplacian = compute_laplacian_spectrum(train_data, nodes_per_graph=train_nodes)
    
    # ========================================================================
    # Load model
    # ========================================================================
    print(f"\n{'-'*80}")
    print("Loading model...")
    print(f"{'-'*80}")
    
    input_dim = train_data.x.shape[1]
    output_dim = train_data.y.shape[1] if train_data.y.dim() > 1 else 1
    edge_dim = train_data.edge_attr.shape[1] if train_data.edge_attr.dim() > 1 else 1
    
    model = load_model(
        args.model_type,
        args.model_path,
        input_dim=input_dim,
        output_dim=output_dim,
        edge_dim=edge_dim,
        device=device
    )
    
    # ========================================================================
    # Evaluate on each test grid
    # ========================================================================
    print(f"\n{'-'*80}")
    print("Evaluating on test grids...")
    print(f"{'-'*80}")
    
    results_list = []
    all_nrmses = []
    all_mmds_degree = []
    all_mmds_laplacian = []
    
    for test_path in test_paths:
        test_grid = get_grid_name(test_path)
        print(f"\n  Testing on: {test_grid}")
        
        # Load test data
        test_data, test_meta = load_data(test_path)
        test_nodes = get_nodes_per_graph(test_data, test_meta)
        print(f"    Nodes: {test_nodes}")
        
        # Run inference
        predictions, targets, mask = run_inference(model, test_data, device)
        print(f"    Predictions shape: {predictions.shape}")
        
        # Compute NRMSE
        nrmse, rmse = compute_nrmse(predictions, targets, mask)
        print(f"    NRMSE: {nrmse:.6f}")
        
        # Get MMDs (from cache or compute)
        if cached_mmds is not None:
            # Look up from cached values
            row = cached_mmds[(cached_mmds['train_grid'] == train_grid) & 
                             (cached_mmds['test_grid'] == test_grid)]
            if len(row) == 0:
                raise ValueError(f"No cached MMD found for {train_grid} → {test_grid}")
            mmd_degree = row['mmd_degree'].values[0]
            mmd_laplacian = row['mmd_laplacian'].values[0]
            print(f"    MMD (degree): {mmd_degree:.6f} [cached]")
            print(f"    MMD (laplacian): {mmd_laplacian:.6f} [cached]")
        else:
            # Compute MMDs
            test_data_cpu = test_data.cpu()
            test_degrees = compute_degree_distribution(test_data_cpu, test_nodes)
            test_laplacian = compute_laplacian_spectrum(test_data_cpu, nodes_per_graph=test_nodes)
            
            mmd_degree = compute_mmd(train_degrees, test_degrees, args.sigma_degree)
            mmd_laplacian = compute_mmd(train_laplacian, test_laplacian, args.sigma_laplacian)
            
            print(f"    MMD (degree): {mmd_degree:.6f}")
            print(f"    MMD (laplacian): {mmd_laplacian:.6f}")
        
        # Collect for g-score
        all_nrmses.append(nrmse)
        all_mmds_degree.append(mmd_degree)
        all_mmds_laplacian.append(mmd_laplacian)
        
        # Store per-grid results
        results_list.append({
            'train_grid': train_grid,
            'test_grid': test_grid,
            'model_type': args.model_type,
            'nrmse': nrmse,
            'rmse': rmse,
            'mmd_degree': mmd_degree,
            'mmd_laplacian': mmd_laplacian,
        })
    
    # ========================================================================
    # Compute G-Score across all test grids
    # ========================================================================
    print(f"\n{'-'*80}")
    print("Computing g-score across all test contexts...")
    print(f"{'-'*80}")
    
    nrmses = np.array(all_nrmses)
    mmds_degree = np.array(all_mmds_degree)
    mmds_laplacian = np.array(all_mmds_laplacian)
    
    print(f"\n  NRMSE values: {nrmses}")
    print(f"  MMD (degree) values: {mmds_degree}")
    print(f"  MMD (laplacian) values: {mmds_laplacian}")
    
    # Use raw g-score (no percentile trimming) since we have few test contexts
    mean_nrmse, std_nrmse, mmd_range_degree, g_score_degree = get_generalization_score_raw(
        mmds_degree, nrmses
    )
    _, _, mmd_range_laplacian, g_score_laplacian = get_generalization_score_raw(
        mmds_laplacian, nrmses
    )
    
    print(f"\n  Mean NRMSE: {mean_nrmse:.6f}")
    print(f"  Std NRMSE: {std_nrmse:.6f}")
    print(f"  MMD range (degree): {mmd_range_degree:.6f}")
    print(f"  MMD range (laplacian): {mmd_range_laplacian:.6f}")
    print(f"\n  ✓ G-Score (Degree): {g_score_degree:.6f}")
    print(f"  ✓ G-Score (Laplacian): {g_score_laplacian:.6f}")
    
    # ========================================================================
    # Save Results
    # ========================================================================
    print(f"\n{'-'*80}")
    print("Saving results...")
    print(f"{'-'*80}")
    
    # Save per-grid results
    results_df = pd.DataFrame(results_list)
    per_grid_file = args.output_file.replace('.csv', '_per_grid.csv')
    results_df.to_csv(per_grid_file, index=False)
    print(f"  ✓ Per-grid results: {per_grid_file}")
    
    # Save summary with g-score
    summary = {
        'train_grid': train_grid,
        'test_grids': ','.join(test_grids),
        'model_type': args.model_type,
        'num_test_grids': len(test_grids),
        'mean_nrmse': mean_nrmse,
        'std_nrmse': std_nrmse,
        'mmd_range_degree': mmd_range_degree,
        'mmd_range_laplacian': mmd_range_laplacian,
        'g_score_degree': g_score_degree,
        'g_score_laplacian': g_score_laplacian,
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(args.output_file, index=False)
    print(f"  ✓ Summary with g-score: {args.output_file}")
    
    # ========================================================================
    # Display Summary
    # ========================================================================
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"\nTrained on: {train_grid}")
    print(f"Tested on: {test_grids}")
    print(f"\nPer-grid results:")
    print(results_df.to_string(index=False))
    print(f"\nG-Score (lower = better generalization):")
    print(f"  Degree-based:    {g_score_degree:.6f}")
    print(f"  Laplacian-based: {g_score_laplacian:.6f}")
    print(f"{'='*80}\n")
    
    return g_score_degree, g_score_laplacian


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
