#!/usr/bin/env python
"""
Batch Evaluation Script for IEEE Transmission Grids
====================================================

Evaluates pre-trained models: train on one IEEE grid, test on the remaining three.
Supports multiple model types and generates comprehensive results.

Usage:
    python batch_eval_ieee.py \
        --data_dir ./ieee-grids \
        --model_dir ./trained_models \
        --models gcn gin gat transformer \
        --training_grid IEEE18 \
        --output_dir ./results

"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
import subprocess

IEEE_GRIDS = ['IEEE18', 'IEEE24', 'IEEE39', 'UK']
MODEL_TYPES = ['gcn', 'gin', 'gat', 'transformer']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Batch evaluate models on IEEE grids'
    )
    parser.add_argument(
        '--data_dir',
        required=True,
        help='Directory containing IEEE grid datasets'
    )
    parser.add_argument(
        '--model_dir',
        required=True,
        help='Directory containing trained model weights'
    )
    parser.add_argument(
        '--training_grid',
        required=True,
        choices=IEEE_GRIDS,
        help='Grid to train on'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['gcn'],
        choices=MODEL_TYPES,
        help='Model types to evaluate'
    )
    parser.add_argument(
        '--output_dir',
        default='./evaluation_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size'
    )
    return parser.parse_args()


def get_test_grids(training_grid):
    """Get remaining grids to test on"""
    return [g for g in IEEE_GRIDS if g != training_grid]


def find_model_weights(model_dir, model_type, training_grid):
    """Find model weights file for given model type and training grid"""
    patterns = [
        f'{training_grid.lower()}_{model_type}.pth',
        f'{training_grid}_{model_type}.pth',
        f'{model_type}_{training_grid.lower()}.pth',
        f'{model_type}_{training_grid}.pth',
    ]
    
    model_path = Path(model_dir)
    for pattern in patterns:
        full_path = model_path / pattern
        if full_path.exists():
            return str(full_path)
    
    # Try to find any matching file
    for pth_file in model_path.glob(f'*{model_type}*{training_grid.lower()}*.pth'):
        return str(pth_file)
    for pth_file in model_path.glob(f'*{model_type}*{training_grid}*.pth'):
        return str(pth_file)
    
    return None


def run_evaluation(data_dir, training_grid, test_grid, model_type, model_path, 
                   batch_size):
    """Run single evaluation using eval_ieee_models.py"""
    
    cmd = [
        'python', 'eval_ieee_models.py',
        '--data_dir', data_dir,
        '--training_grid', training_grid,
        '--test_grid', test_grid,
        '--model_path', model_path,
        '--model_type', model_type,
        '--batch_size', str(batch_size),
        '--output_file', '/tmp/temp_result.csv'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            # Read result from temp file
            result_df = pd.read_csv('/tmp/temp_result.csv')
            return result_df.iloc[0].to_dict()
        else:
            return {
                'training_grid': training_grid,
                'test_grid': test_grid,
                'model_type': model_type,
                'error': f"Evaluation failed: {result.stderr}"
            }
    except subprocess.TimeoutExpired:
        return {
            'training_grid': training_grid,
            'test_grid': test_grid,
            'model_type': model_type,
            'error': "Evaluation timeout (>300s)"
        }
    except Exception as e:
        return {
            'training_grid': training_grid,
            'test_grid': test_grid,
            'model_type': model_type,
            'error': str(e)
        }


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*100}")
    print("IEEE TRANSMISSION GRID BATCH EVALUATION")
    print(f"{'='*100}")
    print(f"Data directory: {args.data_dir}")
    print(f"Model directory: {args.model_dir}")
    print(f"Training grid: {args.training_grid}")
    print(f"Test grids: {get_test_grids(args.training_grid)}")
    print(f"Models to evaluate: {args.models}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*100}\n")
    
    test_grids = get_test_grids(args.training_grid)
    all_results = []
    
    total_evals = len(args.models) * len(test_grids)
    current_eval = 0
    
    for model_type in args.models:
        print(f"\n{'-'*100}")
        print(f"Model: {model_type.upper()}")
        print(f"{'-'*100}")
        
        # Find model weights
        model_path = find_model_weights(args.model_dir, model_type, args.training_grid)
        if not model_path:
            print(f"✗ No model weights found for {model_type} trained on {args.training_grid}")
            print(f"  Looked in: {args.model_dir}")
            continue
        
        print(f"Using model: {model_path}")
        
        for test_grid in test_grids:
            current_eval += 1
            print(f"\n[{current_eval}/{total_evals}] {args.training_grid} → {test_grid}")
            
            result = run_evaluation(
                data_dir=args.data_dir,
                training_grid=args.training_grid,
                test_grid=test_grid,
                model_type=model_type,
                model_path=model_path,
                batch_size=args.batch_size
            )
            
            if 'error' in result and result['error']:
                print(f"  ✗ Error: {result['error']}")
            else:
                print(f"  ✓ NRMSE: {result.get('nrmse_test', 'N/A'):.6f}")
                print(f"  ✓ G-Score (Degree): {result.get('g_score_degree', 'N/A'):.6f}")
                print(f"  ✓ G-Score (Laplacian): {result.get('g_score_laplacian', 'N/A'):.6f}")
            
            all_results.append(result)
    
    # ========================================================================
    # Save Combined Results
    # ========================================================================
    print(f"\n{'='*100}")
    print("SAVING RESULTS")
    print(f"{'='*100}")
    
    results_df = pd.DataFrame(all_results)
    
    # Main results file
    main_results_file = os.path.join(args.output_dir, f'batch_results_{args.training_grid}.csv')
    results_df.to_csv(main_results_file, index=False)
    print(f"✓ Results saved to: {main_results_file}")
    
    # Summary statistics by model
    summary_file = os.path.join(args.output_dir, f'summary_{args.training_grid}.csv')
    
    # Filter out rows with errors
    valid_results = results_df[results_df['error'].isna()] if 'error' in results_df.columns else results_df
    
    if len(valid_results) > 0:
        summary_stats = valid_results.groupby('model_type').agg({
            'nrmse_test': ['mean', 'std', 'min', 'max'],
            'g_score_degree': ['mean', 'std', 'min', 'max'],
            'g_score_laplacian': ['mean', 'std', 'min', 'max'],
            'mmd_degree': ['mean', 'std'],
            'mmd_laplacian': ['mean', 'std']
        })
        summary_stats.to_csv(summary_file)
        print(f"✓ Summary statistics saved to: {summary_file}")
        
        print(f"\n{'-'*100}")
        print("SUMMARY STATISTICS")
        print(f"{'-'*100}")
        print(summary_stats)
    
    print(f"\n{'='*100}\n")


if __name__ == '__main__':
    main()
