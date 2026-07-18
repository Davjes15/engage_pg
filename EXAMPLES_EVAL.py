#!/usr/bin/env python
"""
Example Usage: IEEE Transmission Grid Model Evaluation
=======================================================

This file demonstrates how to use the new evaluation scripts.
Copy and modify for your specific use case.

"""

# ============================================================================
# EXAMPLE 1: Single Evaluation - One Model, One Test Grid
# ============================================================================

"""
Evaluate GCN trained on IEEE18, tested on IEEE24:

    python eval_ieee_models.py \\
      --data_dir ./ieee-grids \\
      --training_grid IEEE18 \\
      --test_grid IEEE24 \\
      --model_path ./trained_models/ieee18_gcn.pth \\
      --model_type gcn \\
      --output_file results_gcn_ieee18_vs_ieee24.csv

Output: Single row CSV with metrics
    training_grid,test_grid,nrmse_test,g_score_degree,...
    IEEE18,IEEE24,0.125,0.142,...
"""

# ============================================================================
# EXAMPLE 2: Batch Evaluation - Multiple Models, Multiple Test Grids
# ============================================================================

"""
Train on IEEE18, evaluate all models against IEEE24, IEEE39, UK:

    python batch_eval_ieee.py \\
      --data_dir ./ieee-grids \\
      --model_dir ./trained_models \\
      --training_grid IEEE18 \\
      --models gcn gin gat transformer \\
      --output_dir ./results_ieee18

Output: 
    - batch_results_IEEE18.csv (12 rows: 4 models × 3 test grids)
    - summary_IEEE18.csv (4 rows: summary statistics per model)
"""

# ============================================================================
# EXAMPLE 3: Python Script - Custom Evaluation Loop
# ============================================================================

import torch
import pandas as pd
from models import GCN, GIN, GAT, TransformerGNN
from training_utils import (
    get_dataloaders, get_device, test, 
    evaluate_mmd, get_generalization_score, setup_pytorch
)
from graph_utils import get_pyg_graphs
import numpy as np

def evaluate_all_ieee_combinations():
    """Evaluate all models on all IEEE grids"""
    
    setup_pytorch()
    device = get_device()
    
    ieee_grids = ['IEEE18', 'IEEE24', 'IEEE39', 'UK']
    models = ['gcn', 'gin', 'gat', 'transformer']
    model_classes = {
        'gcn': GCN,
        'gin': GIN,
        'gat': GAT,
        'transformer': TransformerGNN
    }
    
    results = []
    
    for train_grid in ieee_grids:
        print(f"\n{'='*80}")
        print(f"Training grid: {train_grid}")
        print(f"{'='*80}")
        
        for test_grid in ieee_grids:
            if train_grid == test_grid:
                continue  # Skip same grid
            
            print(f"\nTest grid: {test_grid}")
            
            for model_type in models:
                print(f"  {model_type.upper()}...", end=" ")
                
                try:
                    # Load model
                    train_loader, _, _ = get_dataloaders(
                        data_dir='./ieee-grids',
                        training_grids=[train_grid],
                        batch_size=16
                    )
                    input_dim = next(iter(train_loader)).x.shape[1]
                    
                    model = model_classes[model_type](input_dim=input_dim).to(device)
                    model_path = f'./trained_models/{train_grid.lower()}_{model_type}.pth'
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    
                    # Evaluate
                    _, _, test_loader = get_dataloaders(
                        data_dir='./ieee-grids',
                        training_grids=None,
                        testing_grids=[test_grid],
                        batch_size=16
                    )
                    nrmse_test = test(model, device, test_loader)
                    
                    # Calculate MMD
                    train_graphs = get_pyg_graphs('./ieee-grids', train_grid)
                    test_graphs = get_pyg_graphs('./ieee-grids', test_grid)
                    mmd_deg, mmd_lap = evaluate_mmd(train_graphs, test_graphs)
                    
                    # Calculate g-score
                    m, s, r, g = get_generalization_score(
                        np.array([mmd_deg]), np.array([nrmse_test])
                    )
                    
                    results.append({
                        'train': train_grid,
                        'test': test_grid,
                        'model': model_type,
                        'nrmse': nrmse_test,
                        'mmd': mmd_deg,
                        'g_score': g
                    })
                    
                    print(f"✓ (g-score: {g:.3f})")
                    
                except Exception as e:
                    print(f"✗ ({str(e)[:50]})")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('all_ieee_results.csv', index=False)
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY - Average G-Score by Model")
    print(f"{'='*80}")
    summary = results_df.groupby('model')['g_score'].agg(['mean', 'std', 'min', 'max'])
    print(summary)
    
    return results_df

# Run example
if __name__ == '__main__':
    results = evaluate_all_ieee_combinations()


# ============================================================================
# EXAMPLE 4: Analyzing Results
# ============================================================================

"""
Analyze evaluation results:

    import pandas as pd
    
    # Load results
    results = pd.read_csv('results_ieee18/batch_results_IEEE18.csv')
    
    # Best performing model (lowest g-score)
    best_model = results.loc[results['g_score_degree'].idxmin()]
    print(f"Best model: {best_model['model_type']} with g-score {best_model['g_score_degree']:.3f}")
    
    # Average by model
    print(results.groupby('model_type')['g_score_degree'].mean().sort_values())
    
    # Worst generalization (highest g-score)
    print(results.nlargest(3, 'g_score_degree')[['model_type', 'test_grid', 'g_score_degree']])
    
    # Visualization
    import matplotlib.pyplot as plt
    
    results.pivot_table(
        values='g_score_degree', 
        index='model_type', 
        columns='test_grid'
    ).plot(kind='bar')
    plt.ylabel('G-Score (Degree)')
    plt.xlabel('Model Type')
    plt.title('Generalization Score: Trained on IEEE18')
    plt.tight_layout()
    plt.savefig('generalization_comparison.png')
"""

# ============================================================================
# EXAMPLE 5: Training New Models
# ============================================================================

"""
Train a new model from scratch:

    import torch
    from models import TransformerGNN
    from training_utils import (
        train, get_dataloaders, get_device, 
        setup_pytorch, create_log_dir
    )
    
    setup_pytorch()
    device = get_device()
    
    # Load data
    train_loader, val_loader, _ = get_dataloaders(
        data_dir='./ieee-grids',
        training_grids=['IEEE18'],
        batch_size=32
    )
    
    # Create model
    input_dim = next(iter(train_loader)).x.shape[1]
    model = TransformerGNN(input_dim=input_dim).to(device)
    
    # Train
    print("Training TransformerGNN on IEEE18...")
    train_loss, val_loss, best_loss, train_time, epochs = train(
        model=model,
        device=device,
        loader_train=train_loader,
        loader_val=val_loader,
        epochs=100,
        learning_rate=1e-3,
        early_stopping=True,
        patience=50,
        best_val_weights=True
    )
    
    print(f"Training complete: {train_time:.1f}s, {epochs} epochs")
    
    # Save model
    torch.save(model.state_dict(), './trained_models/ieee18_transformer.pth')
    print("Model saved!")
"""

# ============================================================================
# EXAMPLE 6: Hyperparameter Search - Different Sigma Values
# ============================================================================

"""
Test different MMD kernel bandwidths:

    import subprocess
    
    sigma_values = [1e1, 1e2, 1e3]
    
    for sigma in sigma_values:
        subprocess.run([
            'python', 'eval_ieee_models.py',
            '--data_dir', './ieee-grids',
            '--training_grid', 'IEEE18',
            '--test_grid', 'IEEE24',
            '--model_path', './trained_models/ieee18_gcn.pth',
            '--model_type', 'gcn',
            '--sigma_degree', str(sigma),
            '--output_file', f'results_sigma_{sigma}.csv'
        ])
    
    # Compare results
    import pandas as pd
    for sigma in sigma_values:
        df = pd.read_csv(f'results_sigma_{sigma}.csv')
        print(f"Sigma {sigma}: g-score = {df['g_score_degree'].iloc[0]:.3f}")
"""

# ============================================================================
# EXAMPLE 7: Cross-Validation - Train on Subsets
# ============================================================================

"""
Cross-validation over all 4 grids:

    from itertools import combinations
    import pandas as pd
    
    ieee_grids = ['IEEE18', 'IEEE24', 'IEEE39', 'UK']
    results = []
    
    # Leave-one-out: train on 3, test on 1
    for test_grid in ieee_grids:
        train_grids = [g for g in ieee_grids if g != test_grid]
        
        print(f"Test: {test_grid}, Train: {train_grids}")
        
        # This would require training on multiple grids
        # See cross_context_experiment.py for reference
    
    # Or: train on each pair, test on remaining 2
    for train_grid in ieee_grids:
        test_grids = [g for g in ieee_grids if g != train_grid]
        
        for test_grid in test_grids:
            # Run evaluation
            pass
"""

# ============================================================================
# EXAMPLE 8: Create Results Table
# ============================================================================

"""
Generate publication-ready results table:

    import pandas as pd
    
    # Load batch results
    results = pd.read_csv('results_ieee18/batch_results_IEEE18.csv')
    
    # Pivot table: models vs test grids
    pivot_g_score = results.pivot_table(
        values='g_score_degree',
        index='model_type',
        columns='test_grid',
        aggfunc='mean'
    )
    
    print(pivot_g_score.to_latex())  # LaTeX table for paper
    
    # With bold for best (lowest)
    print("\\\\textbf{G-Score Comparison (Lower is Better)}")
    print(pivot_g_score.round(3))
    
    # Also create NRMSE comparison
    pivot_nrmse = results.pivot_table(
        values='nrmse_test',
        index='model_type',
        columns='test_grid'
    )
    print(pivot_nrmse.round(4))
"""

# ============================================================================
# EXAMPLE 9: Memory-Efficient Batch Processing
# ============================================================================

"""
Process large batches with memory optimization:

    import subprocess
    import time
    
    ieee_grids = ['IEEE18', 'IEEE24', 'IEEE39', 'UK']
    models = ['gcn', 'gin', 'gat', 'transformer']
    
    total = len(ieee_grids) * len(models) * 3  # 3 test grids each
    current = 0
    
    for train_grid in ieee_grids:
        for model_type in models:
            model_path = f'./trained_models/{train_grid.lower()}_{model_type}.pth'
            
            for test_grid in ieee_grids:
                if train_grid == test_grid:
                    continue
                
                current += 1
                print(f"[{current}/{total}] {train_grid} → {test_grid} ({model_type})")
                
                subprocess.run([
                    'python', 'eval_ieee_models.py',
                    '--data_dir', './ieee-grids',
                    '--training_grid', train_grid,
                    '--test_grid', test_grid,
                    '--model_path', model_path,
                    '--model_type', model_type,
                    '--batch_size', '8',  # Smaller batches
                    '--output_file', f'results/{train_grid}_{test_grid}_{model_type}.csv'
                ])
                
                time.sleep(2)  # Cool down between evaluations
"""

# ============================================================================
# EXAMPLE 10: Comparing Against Baseline
# ============================================================================

"""
Compare new models vs traditional baseline (DC-PF):

    from training_utils import get_dataloaders, test_dc_pf, get_device
    
    device = get_device()
    
    test_grids = ['IEEE24', 'IEEE39', 'UK']
    baseline_results = []
    
    for test_grid in test_grids:
        _, _, test_loader = get_dataloaders(
            data_dir='./ieee-grids',
            training_grids=None,
            testing_grids=[test_grid],
            batch_size=16
        )
        
        nrmse_dc = test_dc_pf(device, test_loader)
        
        baseline_results.append({
            'model': 'DC-PowerFlow',
            'test_grid': test_grid,
            'nrmse': nrmse_dc
        })
    
    # Compare with trained models
    trained = pd.read_csv('results_ieee18/batch_results_IEEE18.csv')
    baseline = pd.DataFrame(baseline_results)
    
    print("Trained Models NRMSE:")
    print(trained.groupby('model_type')['nrmse_test'].mean())
    
    print("\\nBaseline (DC-PF) NRMSE:")
    print(baseline['nrmse'].mean())
    
    # Calculate improvement
    improvement = (baseline['nrmse'].mean() - trained['nrmse_test'].mean()) / baseline['nrmse'].mean() * 100
    print(f"\\nImprovement: {improvement:.1f}%")
"""
