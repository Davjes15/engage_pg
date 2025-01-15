import os
import argparse
from itertools import permutations, combinations

import pandas as pd
import numpy as np
from tqdm import tqdm

from models import GCN, ARMA_GNN
from training_utils import (
    create_log_dir,
    setup_logging,
    get_model_save_path,
    setup_pytorch_and_get_device,
    get_dataloaders,
    plot_loss,
    train,
    test,
    test_dc_opf,
    evaluate_mmd
)
from graph_utils import get_dist_grid_codes

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        required=True,
    )
    parser.add_argument(
        "--model",
        default='gcn',
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--save_results",
        action="store_true"
    )
    parser.add_argument(
        "--scenario",
        type=int,
        default=0,
    )
    args = parser.parse_args()
    return args

def get_performance_test_cases(grids_to_compare):
    # List of tuples: (training_grid, testing_grid)
    cases = list(permutations(grids_to_compare, 2))
    return cases

def get_mmd_test_cases(grids_to_compare):
    # List of tuples: (training_grid, testing_grid)
    cases = list(combinations(grids_to_compare, 2))
    return cases

def evaluate_performance(data_dir,
                         model_class,
                         training_grid_code,
                         testing_grid_code,
                         batch_size=16,
                         epochs=1000,
                         add_cycles=False,
                         add_path_lengths=False,
                         log_dir=None,
                         plot=False,
                         save_model=False):
    # Log information about training run
    learning_rate=1e-3
    early_stopping=True
    patience=500
    best_val_weights=True
    print(locals(), flush=True)

    # Create artefacts for logging
    if log_dir:
        setup_logging(log_dir)

    model_weights_path = ''
    if save_model:
        assert log_dir, 'Need to pass a log_dir path in order to save model'
        model_weights_path = get_model_save_path(log_dir)

    # PyTorch setup
    device = setup_pytorch_and_get_device()

    # Get data loaders
    loader_train, loader_val, loader_test = get_dataloaders(
        data_dir=data_dir,
        training_grid=training_grid_code,
        testing_grid=testing_grid_code,
        include_sources=False,
        add_cycles=add_cycles,
        add_path_lengths=add_path_lengths,
        batch_size=batch_size)

    # Create model
    input_dim = next(iter(loader_train)).x.shape[1]
    model = model_class(input_dim=input_dim).to(device)

    # Train the model
    train_loss_vec, val_loss_vec, best_val_loss, train_time, total_epochs = \
        train(model=model,
              device=device,
              loader_train=loader_train,
              loader_val=loader_val,
              epochs=epochs,
              learning_rate=learning_rate,
              early_stopping=early_stopping,
              patience=patience,
              best_val_weights=best_val_weights,
              save_model_to=model_weights_path,
              log_epochs=(log_dir == True))
    
    # Plot the model
    if plot:
        plot_loss(model_class.__name__,
                  train_loss_vec,
                  val_loss_vec,
                  add_cycles,
                  add_path_lengths)

    # Test the model
    nrmse_test = test(model=model,
                      device=device,
                      loader_test=loader_test)
    
    return nrmse_test, best_val_loss, train_time, total_epochs

def evaluate_dc_opf(data_dir, testing_grid_code):
    # Get data loaders
    _, _, loader_test = get_dataloaders(
        data_dir=data_dir,
        training_grid=None,
        testing_grid=testing_grid_code,
        include_sources=True,
        add_cycles=False,
        add_path_lengths=False)
    
    nrmse_test = test_dc_opf(loader_test)
    return nrmse_test

def evaluate_tl_mmd(data_dir,
                    training_grid_code,
                    testing_grid_code,
                    batch_size=16):
    # Get data loaders
    loader_train, loader_val, loader_test = get_dataloaders(
        data_dir=data_dir,
        training_grid=training_grid_code,
        testing_grid=testing_grid_code,
        include_sources=False,
        add_cycles=False,
        add_path_lengths=False,
        batch_size=batch_size)
    
    mmd_degree, mmd_laplacian = evaluate_mmd(
        list(loader_test.dataset) + list(loader_val),
        list(loader_train.dataset)
        )
    
    return mmd_degree, mmd_laplacian

# DATA_DIR = 'outputs/2024-12-30_19:23:46/'

if __name__ == '__main__':
    args = parse_args()

    DATA_DIR = args.data_dir

    # Get model
    model_classes = {'gcn': GCN, 'arma_gnn': ARMA_GNN}
    model_class = model_classes[args.model]

    # Get training params
    batch_size = args.batch_size
    epochs = args.epochs

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
        'training_grid',
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
    for train_grid, test_grid in test_cases:
        for add_cycles, add_path_lengths in variations:
            print(f'\nIteration {i}/{total}')
            nrmse_test, best_val_loss, train_time, total_epochs = \
                evaluate_performance(data_dir=DATA_DIR,
                              model_class=model_class,
                              training_grid_code=train_grid,
                              testing_grid_code=test_grid,
                              batch_size=batch_size,
                              epochs=epochs,
                              add_cycles=add_cycles,
                              add_path_lengths=add_path_lengths)
            performance_results.append(
                (
                    train_grid,
                    test_grid,
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
        nrmse_dc_opf = evaluate_dc_opf(DATA_DIR, test_grid)
        performance_results.append(
            (
                'N/A', # train_grid
                test_grid,
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

    # Compare MMDs between train and test sets, and add to results df
    print('\nCalculating MMDs\n')
    test_cases = get_mmd_test_cases(grids_to_compare)
    results_df['mmd_degree'] = np.nan
    results_df['mmd_laplacian'] = np.nan
    for train_grid, test_grid in tqdm(test_cases):
        mmd_degree, mmd_laplacian = evaluate_tl_mmd(DATA_DIR,
                                                    train_grid,
                                                    test_grid,
                                                    batch_size)
        
        # Since distance is symmetric, check both directions.
        results_df.loc[
            (
                (results_df['training_grid'] == train_grid) &
                (results_df['testing_grid'] == test_grid)
            ) |
            (
                (results_df['training_grid'] == test_grid) &
                (results_df['testing_grid'] == train_grid)
            ),
            ['mmd_degree', 'mmd_laplacian']] = mmd_degree, mmd_laplacian

    # Save all results
    if args.save_results:
        log_dir = create_log_dir(model_class.__name__)
        results_file = os.path.join(log_dir, 'results.csv')
        results_df.to_csv(results_file)
        print(f'\nTransfer learning results saved to: {results_file}')