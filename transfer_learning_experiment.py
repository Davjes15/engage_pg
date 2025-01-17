import os
import argparse
from itertools import permutations, combinations

import pandas as pd
import numpy as np
from tqdm import tqdm

from models import GCN, ARMA_GNN
from training_utils import (
    create_log_dir,
    setup_file_output,
    get_model_save_path,
    setup_pytorch,
    get_device,
    get_dataloaders,
    plot_loss,
    train,
    test,
    test_dc_pf,
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
        "--scenario",
        type=int,
        default=1,
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
        "--plot",
        action="store_true"
    )
    parser.add_argument(
        "--compute_mmd",
        action="store_true"
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
                         training_grid_codes,
                         testing_grid_codes,
                         batch_size=16,
                         epochs=1000,
                         shuffle=False,
                         add_cycles=False,
                         add_path_lengths=False,
                         log_dir=None,
                         plot=False,
                         save_model=False,
                         set_file_output=False,
                         experiment_id=0):
    # Log information about training run
    learning_rate=1e-3
    early_stopping=True
    patience=500
    best_val_weights=True
    print(locals(), flush=True)

    # Create artefacts for logging
    if set_file_output:
        setup_file_output(log_dir)

    model_weights_path = ''
    if save_model or plot:
        assert log_dir, 'Need to pass a log_dir path in order to save model or plot loss'
    if save_model:
        model_weights_path = get_model_save_path(log_dir)

    # PyTorch setup
    device = get_device()
    print(f"Training using {device}", flush=True)

    # Get data loaders
    loader_train, loader_val, loader_test = get_dataloaders(
        data_dir=data_dir,
        training_grids=training_grid_codes,
        testing_grids=testing_grid_codes,
        init_dc=True,
        add_cycles=add_cycles,
        add_path_lengths=add_path_lengths,
        batch_size=batch_size,
        shuffle=shuffle)

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
        plot_loss(log_dir,
                  model_class.__name__,
                  train_loss_vec,
                  val_loss_vec,
                  add_cycles,
                  add_path_lengths,
                  fig_id=experiment_id)

    # Test the model
    nrmse_test = test(model=model,
                      device=device,
                      loader_test=loader_test)
    
    return nrmse_test, best_val_loss, train_time, total_epochs

def evaluate_dc_pf(data_dir, testing_grid_code):
    # Get data loaders
    _, _, loader_test = get_dataloaders(
        data_dir=data_dir,
        training_grids=None,
        testing_grids=[testing_grid_code],
        add_cycles=False,
        add_path_lengths=False)
    
    device = get_device()
    nrmse_test = test_dc_pf(device, loader_test)
    return nrmse_test

def evaluate_tl_mmd(data_dir,
                    training_grid_codes,
                    testing_grid_codes):
    # Get data loaders
    loader_train, loader_val, loader_test = get_dataloaders(
        data_dir=data_dir,
        training_grids=training_grid_codes,
        testing_grids=testing_grid_codes,
        add_cycles=False,
        add_path_lengths=False)
    
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
        'training_grid',
        'testing_grid',
        'cycles',
        'path_lengths',
        'nrmse_test',
        'best_val_loss',
        'train_time',
        'total_epochs',
        'dc_pf'
    ]
    performance_results = []
    i = 1
    total = len(test_cases)*len(variations)
    print('\nCalculating Performance Metrics')
    for train_grid, test_grid in test_cases:
        for add_cycles, add_path_lengths in variations:
            print(f'\nIteration {i}/{total}')
            nrmse_test, best_val_loss, train_time, total_epochs = \
                evaluate_performance(data_dir=DATA_DIR,
                              model_class=model_class,
                              training_grid_codes=[train_grid],
                              testing_grid_codes=[test_grid],
                              batch_size=batch_size,
                              epochs=epochs,
                              shuffle=False,
                              add_cycles=add_cycles,
                              add_path_lengths=add_path_lengths,
                              log_dir=log_dir,
                              plot=plot,
                              experiment_id=i)
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
                    False # dc_pf
                )
            )
            print(f'\tBest val loss: {best_val_loss}\n\tNRMSE: {nrmse_test}')
            i += 1
        print('\nEvaluating dc opf...')
        nrmse_dc_pf = evaluate_dc_pf(DATA_DIR, test_grid)
        print('...complete')
        performance_results.append(
            (
                'N/A', # train_grid
                test_grid,
                False, # add_cycles
                False, # add_path_lengths
                nrmse_dc_pf,
                np.nan, # best_val_loss
                np.nan, # train_time
                np.nan, # total_epochs
                True # dc_pf
            )
        )
    results_df = pd.DataFrame(performance_results, columns=column_names)

    # Save all results intermediately before we do next step.
    results_file = None
    if log_dir:
        results_file = os.path.join(log_dir, 'results_tl.csv')
    if save_results and results_file:
        results_df.to_csv(results_file)
        print(f'\nSaved TL results to: {results_file}')

    if args.compute_mmd:
        # Compare MMDs between train and test sets, and add to results df
        print('Calculating MMDs...')
        test_cases = get_mmd_test_cases(grids_to_compare)
        results_df['mmd_degree'] = np.nan
        results_df['mmd_laplacian'] = np.nan
        for train_grid, test_grid in tqdm(test_cases):
            mmd_degree, mmd_laplacian = \
                evaluate_tl_mmd(data_dir=DATA_DIR,
                                training_grid_codes=[train_grid],
                                testing_grid_codes=[test_grid])

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

        # Save the rest of the results with mmd
        if save_results and results_file:
            results_df.to_csv(results_file)
            print(f'Saved TL results w/ mmd to: {results_file}')

    print("\nTraining complete")