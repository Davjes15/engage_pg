#!/usr/bin/env python
"""
IEEE Transmission Grid Model Evaluation Script
==============================================

Evaluate pre-trained models (GCN, GIN, GAT, Transformer) on multiple IEEE transmission grids.
Trains on one grid and tests on the remaining three grids, computing g-score for each test.

Usage:
    # Single evaluation
    python eval_ieee_models.py \
        --data_dir ./ieee-grids \
        --training_grid IEEE18 \
        --test_grid IEEE24 \
        --model_path ./models/ieee18_gcn.pth \
        --model_type gcn

    # Batch evaluation script (see bottom for example)

"""

import argparse
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

from models import GCN, GIN, GAT, TransformerGNN
from training_utils import (
    get_dataloaders,
    get_device,
    test,
    evaluate_mmd,
    get_generalization_score,
    setup_pytorch
)
from graph_utils import get_pyg_graphs


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate pre-trained models on IEEE transmission grids'
    )
    parser.add_argument(
        '--data_dir',
        required=True,
        help='Base directory containing grid datasets (IEEE18, IEEE24, IEEE39, UK)'
    )
    parser.add_argument(
        '--training_grid',
        required=True,
        help='Training grid name (IEEE18, IEEE24, IEEE39, or UK)'
    )
    parser.add_argument(
        '--test_grid',
        required=True,
        help='Test grid name (IEEE18, IEEE24, IEEE39, or UK)'
    )
    parser.add_argument(
        '--model_path',
        required=True,
        help='Path to trained model weights (.pth file)'
    )
    parser.add_argument(
        '--model_type',
        default='gcn',
        choices=['gcn', 'gin', 'gat', 'transformer'],
        help='Type of model to load'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for evaluation'
    )

    parser.add_argument(
        '--output_file',
        default='evaluation_results.csv',
        help='Output CSV file for results'
    )
    parser.add_argument(
        '--sigma_degree',
        type=float,
        default=1e2,
        help='Sigma parameter for degree distribution MMD'
    )
    parser.add_argument(
        '--sigma_laplacian',
        type=float,
        default=1e-2,
        help='Sigma parameter for Laplacian spectrum MMD'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed information'
    )
    
    return parser.parse_args()


def load_model(model_type, model_path, input_dim, device):
    """Load model from .pth file"""
    model_classes = {
        'gcn': GCN,
        'gin': GIN,
        'gat': GAT,
        'transformer': TransformerGNN
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create model
    model = model_classes[model_type](input_dim=input_dim).to(device)
    
    # Load weights
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model from {model_path}: {e}")


def evaluate_pair(data_dir, training_grid, test_grid, model, device, batch_size,
                  sigma_degree, sigma_laplacian):
    """Evaluate model on a single train-test pair"""
    
    results = {
        'training_grid': training_grid,
        'test_grid': test_grid,
        'nrmse_test': None,
        'mmd_degree': None,
        'mmd_laplacian': None,
        'mean_nrmse': None,
        'std_nrmse': None,
        'mmd_range_degree': None,
        'mmd_range_laplacian': None,
        'g_score_degree': None,
        'g_score_laplacian': None,
        'error': None
    }
    
    try:
        # Load test data
        _, _, test_loader = get_dataloaders(
            data_dir=data_dir,
            training_grids=None,
            testing_grids=[test_grid],
            batch_size=batch_size
        )
        
        # Evaluate
        nrmse_test = test(model=model, device=device, loader_test=test_loader)
        results['nrmse_test'] = nrmse_test
        
        # Load graphs for MMD calculation
        training_graphs = get_pyg_graphs(data_dir, training_grid, split='train')
        test_graphs = get_pyg_graphs(data_dir, test_grid, split='train')
        
        # Calculate MMD
        mmd_degree, mmd_laplacian = evaluate_mmd(
            training_dataset=training_graphs,
            testing_dataset=test_graphs,
            sigma_degree=sigma_degree,
            sigma_laplacian=sigma_laplacian
        )
        results['mmd_degree'] = mmd_degree
        results['mmd_laplacian'] = mmd_laplacian
        
        # Calculate g-score
        nrmses = np.array([nrmse_test])
        mmds_degree = np.array([mmd_degree])
        mmds_laplacian = np.array([mmd_laplacian])
        
        mean_nrmse, std_nrmse, mmd_range_degree, g_score_degree = get_generalization_score(
            mmds_degree, nrmses
        )
        _, _, mmd_range_laplacian, g_score_laplacian = get_generalization_score(
            mmds_laplacian, nrmses
        )
        
        results['mean_nrmse'] = mean_nrmse
        results['std_nrmse'] = std_nrmse
        results['mmd_range_degree'] = mmd_range_degree
        results['mmd_range_laplacian'] = mmd_range_laplacian
        results['g_score_degree'] = g_score_degree
        results['g_score_laplacian'] = g_score_laplacian
        
    except Exception as e:
        results['error'] = str(e)
    
    return results


def main():
    args = parse_args()
    
    # Setup
    setup_pytorch()
    device = get_device()
    
    print(f"\n{'='*80}")
    print("IEEE TRANSMISSION GRID MODEL EVALUATION")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Data directory: {args.data_dir}")
    print(f"Training grid: {args.training_grid}")
    print(f"Test grid: {args.test_grid}")
    print(f"Model type: {args.model_type}")
    print(f"Model path: {args.model_path}")
    
    # ========================================================================
    # Load Model
    # ========================================================================
    print(f"\n{'-'*80}")
    print("Loading model...")
    print(f"{'-'*80}")
    
    # Get input dimension from training data
    try:
        train_loader, _, _ = get_dataloaders(
            data_dir=args.data_dir,
            training_grids=[args.training_grid],
            testing_grids=None,
            batch_size=args.batch_size
        )
        input_dim = next(iter(train_loader)).x.shape[1]
        print(f"Input dimension: {input_dim}")
    except Exception as e:
        print(f"✗ Error loading training data: {e}")
        return
    
    try:
        model = load_model(args.model_type, args.model_path, input_dim, device)
        print(f"✓ Model loaded successfully from {args.model_path}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # ========================================================================
    # Evaluate
    # ========================================================================
    print(f"\n{'-'*80}")
    print(f"Evaluating {args.model_type.upper()} trained on {args.training_grid} → tested on {args.test_grid}")
    print(f"{'-'*80}")
    
    results = evaluate_pair(
        data_dir=args.data_dir,
        training_grid=args.training_grid,
        test_grid=args.test_grid,
        model=model,
        device=device,
        batch_size=args.batch_size,
        sigma_degree=args.sigma_degree,
        sigma_laplacian=args.sigma_laplacian
    )
    
    if results['error']:
        print(f"✗ Evaluation failed: {results['error']}")
        return
    
    # ========================================================================
    # Display Results
    # ========================================================================
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"Test NRMSE: {results['nrmse_test']:.6f}")
    print(f"MMD Degree Distribution: {results['mmd_degree']:.6f}")
    print(f"MMD Laplacian Spectrum: {results['mmd_laplacian']:.6f}")
    print(f"{'-'*80}")
    print(f"Mean NRMSE: {results['mean_nrmse']:.6f}")
    print(f"Std NRMSE: {results['std_nrmse']:.6f}")
    print(f"G-SCORE (Degree): {results['g_score_degree']:.6f}")
    print(f"G-SCORE (Laplacian): {results['g_score_laplacian']:.6f}")
    print(f"{'='*80}\n")
    
    # Save results
    results_df = pd.DataFrame([results])
    try:
        results_df.to_csv(args.output_file, index=False)
        print(f"✓ Results saved to: {args.output_file}")
    except Exception as e:
        print(f"✗ Error saving results: {e}")


if __name__ == '__main__':
    main()
