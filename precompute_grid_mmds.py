#!/usr/bin/env python
"""
Precompute topological features and pairwise MMDs for all grids.
=========================================================================

This script computes degree distributions, Laplacian spectra, and pairwise MMDs
for all grids once. These values are grid-dependent (not model-dependent) and
can be reused across all model evaluations.

Usage:
    python precompute_grid_mmds.py --grids_dir ./ieee-grids --output_dir ./cached_mmds

Output files:
    - grid_features.pt: Topological features (degrees, laplacian) for each grid
    - pairwise_mmds.csv: MMD values between all grid pairs
"""

import argparse
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import warnings
from itertools import combinations

warnings.filterwarnings('ignore')

# Add ENGAGE's ggme module to path for MMD functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ggme', 'src'))
from metrics.mmd import mmd as engage_mmd
from metrics.kernels import gaussian_kernel


def parse_args():
    parser = argparse.ArgumentParser(
        description='Precompute topological features and MMDs for all grids'
    )
    parser.add_argument(
        '--grids_dir',
        default='./ieee-grids',
        help='Directory containing grid folders'
    )
    parser.add_argument(
        '--output_dir',
        default='./cached_mmds',
        help='Directory to save cached features and MMDs'
    )
    parser.add_argument(
        '--sigma_degree',
        type=float,
        default=1e2,
        help='Sigma parameter for degree distribution MMD (larger avoids saturation)'
    )
    parser.add_argument(
        '--sigma_laplacian',
        type=float,
        default=1e-2,
        help='Sigma parameter for Laplacian spectrum MMD'
    )
    parser.add_argument(
        '--k_laplacian',
        type=int,
        default=10,
        help='Number of Laplacian eigenvalues to compute'
    )
    parser.add_argument(
        '--features',
        choices=['degree', 'laplacian', 'both'],
        default='both',
        help='Which topological features to compute MMD for: degree, laplacian, or both'
    )
    
    return parser.parse_args()


def load_data(data_path):
    """Load PowerGraph-Node batched data"""
    data_tuple = torch.load(data_path, weights_only=False)
    
    if isinstance(data_tuple, tuple) and len(data_tuple) == 2:
        graph_data, metadata = data_tuple
    else:
        graph_data = data_tuple
        metadata = None
    
    return graph_data, metadata


def get_grid_name(data_path):
    """Extract grid name from data path"""
    path = Path(data_path)
    if path.parent.name == 'processed_node':
        return path.parent.parent.name
    return path.parent.name


def get_nodes_per_graph(data, metadata):
    """Extract nodes per graph from metadata"""
    if metadata is not None and 'x' in metadata:
        x_idx = metadata['x']
        if len(x_idx) > 1:
            return (x_idx[1] - x_idx[0]).item()
    return None


def discover_all_grids(grids_dir):
    """Find all grids in directory"""
    grid_paths = {}
    
    for grid_dir in Path(grids_dir).iterdir():
        if grid_dir.is_dir():
            data_file = grid_dir / 'processed_node' / 'data.pt'
            if data_file.exists():
                grid_paths[grid_dir.name] = str(data_file)
    
    return grid_paths


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
    
    # Get local edge indices (just take edges from first sample - topology is shared)
    edge_idx_local = edge_index[0] % n if nodes_per_graph else edge_index[0]
    
    # Only use edges within the first sample's topology
    # (edges repeat for each sample, so we only need the unique ones)
    unique_edges = torch.unique(edge_idx_local[edge_idx_local < n])
    
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
        return np.zeros((1, k))
    
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
    """Compute MMD using ENGAGE's unbiased estimator."""
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


def main():
    args = parse_args()
    
    print(f"\n{'='*80}")
    print("PRECOMPUTING GRID TOPOLOGICAL FEATURES AND PAIRWISE MMDs")
    print(f"{'='*80}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Discover all grids
    grid_paths = discover_all_grids(args.grids_dir)
    grid_names = sorted(grid_paths.keys())
    
    print(f"\nFound {len(grid_names)} grids: {grid_names}")
    
    # ========================================================================
    # Compute topological features for each grid
    # ========================================================================
    print(f"\n{'-'*80}")
    print("Computing topological features...")
    print(f"{'-'*80}")
    
    features = {}
    
    for grid_name in grid_names:
        data_path = grid_paths[grid_name]
        print(f"\n  {grid_name}:")
        
        data, metadata = load_data(data_path)
        nodes_per_graph = get_nodes_per_graph(data, metadata)
        
        print(f"    Nodes: {nodes_per_graph}")
        
        # Compute features based on --features flag
        feature_dict = {'nodes_per_graph': nodes_per_graph}
        
        if args.features in ['degree', 'both']:
            degrees = compute_degree_distribution(data, nodes_per_graph)
            feature_dict['degree_distribution'] = degrees
            print(f"    Degree shape: {degrees.shape}, mean: {degrees.mean():.2f}")
        
        if args.features in ['laplacian', 'both']:
            laplacian = compute_laplacian_spectrum(data, k=args.k_laplacian, nodes_per_graph=nodes_per_graph)
            feature_dict['laplacian_spectrum'] = laplacian
            print(f"    Laplacian shape: {laplacian.shape}")
        
        features[grid_name] = feature_dict
    
    # Save features
    features_file = os.path.join(args.output_dir, 'grid_features.pt')
    torch.save(features, features_file)
    print(f"\n✓ Saved grid features to: {features_file}")
    
    # ========================================================================
    # Compute pairwise MMDs
    # ========================================================================
    print(f"\n{'-'*80}")
    print("Computing pairwise MMDs...")
    print(f"{'-'*80}")
    
    mmd_results = []
    
    compute_degree = args.features in ['degree', 'both']
    compute_laplacian = args.features in ['laplacian', 'both']
    
    print(f"  Computing: {'degree ' if compute_degree else ''}{'laplacian' if compute_laplacian else ''}")
    
    # Compute all pairs (including A→B and B→A for convenience)
    for grid_a in grid_names:
        for grid_b in grid_names:
            mmd_degree = None
            mmd_laplacian = None
            
            if grid_a == grid_b:
                if compute_degree:
                    mmd_degree = 0.0
                if compute_laplacian:
                    mmd_laplacian = 0.0
            else:
                feat_a = features[grid_a]
                feat_b = features[grid_b]
                
                if compute_degree:
                    mmd_degree = compute_mmd(
                        feat_a['degree_distribution'],
                        feat_b['degree_distribution'],
                        args.sigma_degree
                    )
                if compute_laplacian:
                    mmd_laplacian = compute_mmd(
                        feat_a['laplacian_spectrum'],
                        feat_b['laplacian_spectrum'],
                        args.sigma_laplacian
                    )
            
            result = {
                'train_grid': grid_a,
                'test_grid': grid_b,
            }
            if compute_degree:
                result['mmd_degree'] = mmd_degree
            if compute_laplacian:
                result['mmd_laplacian'] = mmd_laplacian
            
            mmd_results.append(result)
            
            if grid_a != grid_b:
                parts = []
                if compute_degree:
                    parts.append(f"MMD_deg={mmd_degree:.6f}")
                if compute_laplacian:
                    parts.append(f"MMD_lap={mmd_laplacian:.6f}")
                print(f"  {grid_a} → {grid_b}: {', '.join(parts)}")
    
    # Save MMDs
    mmd_df = pd.DataFrame(mmd_results)
    mmd_file = os.path.join(args.output_dir, 'pairwise_mmds.csv')
    mmd_df.to_csv(mmd_file, index=False)
    print(f"\n✓ Saved pairwise MMDs to: {mmd_file}")
    
    # ========================================================================
    # Display summary table
    # ========================================================================
    if compute_degree:
        print(f"\n{'='*80}")
        print("PAIRWISE MMD SUMMARY (Degree-based)")
        print(f"{'='*80}")
        pivot = mmd_df.pivot(index='train_grid', columns='test_grid', values='mmd_degree')
        print(pivot.to_string(float_format=lambda x: f'{x:.4f}'))
    
    if compute_laplacian:
        print(f"\n{'='*80}")
        print("PAIRWISE MMD SUMMARY (Laplacian-based)")
        print(f"{'='*80}")
        pivot_lap = mmd_df.pivot(index='train_grid', columns='test_grid', values='mmd_laplacian')
        print(pivot_lap.to_string(float_format=lambda x: f'{x:.4f}'))
    
    print(f"\n{'='*80}")
    print("DONE! Use --cached_mmds in eval_powergraph_models.py to skip MMD computation")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
