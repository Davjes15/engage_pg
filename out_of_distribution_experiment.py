import os
import argparse
from itertools import permutations, combinations

import pandas as pd
import numpy as np
from tqdm import tqdm

from models import GCN, ARMA_GNN
from training_utils import (
    create_log_dir,
    setup_pytorch,
)
from graph_utils import get_dist_grid_codes
from transfer_learning_experiment import (
    parse_args,
    evaluate_performance,
    evaluate_dc_opf,
    evaluate_tl_mmd
)

def get_performance_test_cases(grids_to_compare):
    cases = []
    for i in range(len(grids_to_compare)):
        target_grid = grids_to_compare[i]
        training_grids = grids_to_compare[:i] + grids_to_compare[i+1:]
        cases.append([training_grids, target_grid])
    
    # List of tuples: ([training_grids], testing_grid)
    return cases

def get_mmd_test_cases(grids_to_compare):
    return get_performance_test_cases(grids_to_compare)

if __name__ == '__main__':
    args = parse_args()

    DATA_DIR = args.data_dir

    # Get model
    model_classes = {'gcn': GCN, 'arma_gnn': ARMA_GNN}
    model_class = model_classes[args.model]

    # Set up pytorch and training
    setup_pytorch()
    batch_size = args.batch_size
    epochs = args.epochs
    save_results = args.save_results
    plot = args.plot
    log_dir = None
    if save_results or plot:
        log_dir = create_log_dir(model_class.__name__)

    # Grids to compare pairwise
    grids_to_compare = get_dist_grid_codes(scenario=args.scenario)
    test_cases = get_performance_test_cases(grids_to_compare)
    
    # Variations of each pairwise comparison (add_cycles, add_path_lengths)
    variations = [(False, False),
                  (True, True),
                  (True, False),
                  (False, True)]

    # Run the training and test for every combination, and save results in df
    column_names = [
        'testing_grid',
        'cycles',
        'path_lengths',
        'nrmse_test',
        'best_val_loss',
        'train_time',
        'total_epochs',
        'dc_opf'
    ]
    performance_results = []
    i = 1
    total = len(test_cases)*len(variations)
    print('\nCalculating Performance Metrics\n')
    for train_grids, target in test_cases:
        for add_cycles, add_path_lengths in variations:
            print(f'\nIteration {i}/{total}')
            nrmse_test, best_val_loss, train_time, total_epochs = \
                evaluate_performance(data_dir=DATA_DIR,
                              model_class=model_class,
                              training_grid_codes=train_grids,
                              testing_grid_codes=[target],
                              batch_size=batch_size,
                              epochs=epochs,
                              shuffle=True,
                              add_cycles=add_cycles,
                              add_path_lengths=add_path_lengths,
                              log_dir=log_dir,
                              plot=plot,
                              experiment_id=i-1)
            performance_results.append(
                (
                    target,
                    add_cycles,
                    add_path_lengths,
                    nrmse_test,
                    best_val_loss,
                    train_time,
                    total_epochs,
                    False # dc_opf
                )
            )
            i += 1
        print('Solving using dc opf')
        nrmse_dc_opf = evaluate_dc_opf(DATA_DIR, target)
        performance_results.append(
            (
                target,
                False, # add_cycles
                False, # add_path_lengths
                nrmse_dc_opf,
                np.nan, # best_val_loss
                np.nan, # train_time
                np.nan, # total_epochs
                True # dc_opf
            )
        )
    results_df = pd.DataFrame(performance_results, columns=column_names)

    # Save all results intermediately before we do next step.
    results_file = None
    if log_dir:
        results_file = os.path.join(log_dir, 'results.csv')
    if save_results and results_file:
        results_df.to_csv(results_file)
        print(f'\nSaving intermediate results to: {results_file}')

    # Compare MMDs between train and test sets, and add to results df
    print('Calculating MMDs...')
    test_cases = get_mmd_test_cases(grids_to_compare)
    results_df['mmd_degree'] = np.nan
    results_df['mmd_laplacian'] = np.nan
    for train_grids, target in tqdm(test_cases):
        mmd_degree, mmd_laplacian = \
            evaluate_tl_mmd(data_dir=DATA_DIR,
                            training_grid_codes=train_grids,
                            testing_grid_codes=[target])
        
        # Since distance is symmetric, check both directions.
        results_df.loc[
                results_df['testing_grid'] == target,
                ['mmd_degree', 'mmd_laplacian']
            ] = mmd_degree, mmd_laplacian

    # Save the rest of the results with mmd
    if save_results and results_file:
        results_df.to_csv(results_file)
        print(f'Transfer learning results w/ mmd saved to: {results_file}')

    print("\nTraining complete")