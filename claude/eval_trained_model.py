#!/usr/bin/env python
"""
Custom Model Evaluation Script
==============================

Evaluate a pre-trained model on a new test dataset and compute g-score.

Usage:
    python eval_trained_model.py \
        --data_dir path/to/data \
        --training_grid grid_name_1 \
        --test_grid grid_name_2 \
        --model_path path/to/model_weights \
        --model_type gcn

"""

import argparse
import torch
import numpy as np
import pandas as pd
from models import GCN, ARMA_GNN
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
        description='Evaluate pre-trained model and compute g-score'
    )
    parser.add_argument(
        '--data_dir',
        required=True,
        help='Base directory containing grid datasets'
    )
    parser.add_argument(
        '--training_grid',
        required=True,
        help='Name of training grid directory'
    )
    parser.add_argument(
        '--test_grid',
        required=True,
        help='Name of test grid directory'
    )
    parser.add_argument(
        '--model_path',
        required=True,
        help='Path to trained model weights file'
    )
    parser.add_argument(
        '--model_type',
        default='gcn',
        choices=['gcn', 'arma_gnn'],
        help='Type of model to load'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--add_cycles',
        action='store_true',
        help='Include cycle features'
    )
    parser.add_argument(
        '--add_path_lengths',
        action='store_true',
        help='Include path length features'
    )
    parser.add_argument(
        '--add_degree',
        action='store_true',
        help='Include degree features'
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
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    setup_pytorch()
    device = get_device()
    print(f"\n{'='*70}")
    print("CUSTOM MODEL EVALUATION")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Data directory: {args.data_dir}")
    print(f"Training grid: {args.training_grid}")
    print(f"Test grid: {args.test_grid}")
    print(f"Model type: {args.model_type}")
    print(f"Model path: {args.model_path}")
    
    # ========================================================================
    # Load Model
    # ========================================================================
    print(f"\n{'-'*70}")
    print("Loading model...")
    print(f"{'-'*70}")
    
    # Get input dimension from training data
    train_loader, _, _ = get_dataloaders(
        data_dir=args.data_dir,
        training_grids=[args.training_grid],
        testing_grids=None,
        batch_size=args.batch_size,
        add_cycles=args.add_cycles,
        add_path_lengths=args.add_path_lengths,
        add_degree=args.add_degree
    )
    
    input_dim = next(iter(train_loader)).x.shape[1]
    print(f"Input dimension: {input_dim}")
    
    # Create and load model
    model_classes = {'gcn': GCN, 'arma_gnn': ARMA_GNN}
    model = model_classes[args.model_type](input_dim=input_dim).to(device)
    
    try:
        model.load_state_dict(
            torch.load(args.model_path, weights_only=True, map_location=device)
        )
        print(f"✓ Model loaded successfully")
    except FileNotFoundError:
        print(f"✗ Error: Model file not found at {args.model_path}")
        return
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # ========================================================================
    # Evaluate on Test Data
    # ========================================================================
    print(f"\n{'-'*70}")
    print("Evaluating on test set...")
    print(f"{'-'*70}")
    
    try:
        _, _, test_loader = get_dataloaders(
            data_dir=args.data_dir,
            training_grids=None,
            testing_grids=[args.test_grid],
            batch_size=args.batch_size,
            add_cycles=args.add_cycles,
            add_path_lengths=args.add_path_lengths,
            add_degree=args.add_degree
        )
    except Exception as e:
        print(f"✗ Error loading test data: {e}")
        return
    
    nrmse_test = test(model=model, device=device, loader_test=test_loader)
    print(f"✓ Test NRMSE: {nrmse_test:.6f}")
    
    # ========================================================================
    # Calculate MMD (Graph Dissimilarity)
    # ========================================================================
    print(f"\n{'-'*70}")
    print("Calculating MMD between training and test graphs...")
    print(f"{'-'*70}")
    
    try:
        training_graphs = get_pyg_graphs(args.data_dir, args.training_grid, split='train')
        test_graphs = get_pyg_graphs(args.data_dir, args.test_grid, split='train')
        print(f"Training graphs: {len(training_graphs)}")
        print(f"Test graphs: {len(test_graphs)}")
    except Exception as e:
        print(f"✗ Error loading graphs: {e}")
        return
    
    try:
        mmd_degree, mmd_laplacian = evaluate_mmd(
            training_dataset=training_graphs,
            testing_dataset=test_graphs,
            sigma_degree=args.sigma_degree,
            sigma_laplacian=args.sigma_laplacian
        )
        print(f"✓ MMD Degree Distribution: {mmd_degree:.6f}")
        print(f"✓ MMD Laplacian Spectrum: {mmd_laplacian:.6f}")
    except Exception as e:
        print(f"✗ Error calculating MMD: {e}")
        return
    
    # ========================================================================
    # Calculate G-Score
    # ========================================================================
    print(f"\n{'-'*70}")
    print("Calculating Generalization Score...")
    print(f"{'-'*70}")
    
    # Create arrays for g-score calculation
    nrmses = np.array([nrmse_test])
    mmds_degree = np.array([mmd_degree])
    mmds_laplacian = np.array([mmd_laplacian])
    
    try:
        mean_nrmse, std_nrmse, mmd_range_degree, g_score_degree = get_generalization_score(
            mmds_degree, nrmses
        )
        _, _, mmd_range_laplacian, g_score_laplacian = get_generalization_score(
            mmds_laplacian, nrmses
        )
        print(f"✓ G-Score calculated")
    except Exception as e:
        print(f"✗ Error calculating g-score: {e}")
        return
    
    # ========================================================================
    # Results Summary
    # ========================================================================
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Training Grid: {args.training_grid}")
    print(f"Test Grid: {args.test_grid}")
    print(f"Model Type: {args.model_type}")
    print(f"{'-'*70}")
    print(f"Test NRMSE: {nrmse_test:.6f}")
    print(f"MMD Degree: {mmd_degree:.6f}")
    print(f"MMD Laplacian: {mmd_laplacian:.6f}")
    print(f"{'-'*70}")
    print(f"Mean NRMSE: {mean_nrmse:.6f}")
    print(f"Std NRMSE: {std_nrmse:.6f}")
    print(f"MMD Range (Degree): {mmd_range_degree:.6f}")
    print(f"MMD Range (Laplacian): {mmd_range_laplacian:.6f}")
    print(f"{'-'*70}")
    print(f"G-SCORE (Degree): {g_score_degree:.6f}")
    print(f"G-SCORE (Laplacian): {g_score_laplacian:.6f}")
    print(f"{'='*70}\n")
    
    # Save results
    results_df = pd.DataFrame({
        'training_grid': [args.training_grid],
        'test_grid': [args.test_grid],
        'model_type': [args.model_type],
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
    
    try:
        results_df.to_csv(args.output_file, index=False)
        print(f"✓ Results saved to: {args.output_file}")
    except Exception as e:
        print(f"✗ Error saving results: {e}")


if __name__ == '__main__':
    main()
