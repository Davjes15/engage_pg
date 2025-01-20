import os

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
    get_model_save_path,
    parse_args,
    evaluate_performance,
    evaluate_dc_pf,
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
    save_model = args.save_model
    eval_only = args.eval_only
    load_model_dir = args.load_model_dir
    log_dir = None
    if save_results or plot:
        log_dir = create_log_dir(model_class.__name__)

    grids_to_compare = get_dist_grid_codes(scenario=args.scenario)
    if not args.skip_experiment:
        # Grids to compare pairwise
        grids_to_compare = get_dist_grid_codes(scenario=args.scenario)
        test_cases = get_performance_test_cases(grids_to_compare)

        # Variations of each pairwise comparison (add_cycles, add_path_lengths, add_degree)
        variations = [(False, False, False), # none
                    (True, True, True), # all
                    (True, True, False), # cycles + path lengths
                    (True, False, False), # cycles
                    (False, True, False), # path lengths
                    (False, False, True)] # degree

        # Run the training and test for every combination, and save results in df
        column_names = [
            'testing_grid',
            'cycles',
            'path_lengths',
            'degree',
            'nrmse_test',
            'nrmse_range', 'nrmse_mean', 'nrmse_std',
            'best_val_loss',
            'train_time',
            'total_epochs'
        ]
        results = []
        i = 1
        total = len(test_cases)*len(variations)
        print('\nCalculating Performance Metrics')
        for train_grids, target in test_cases:
            for add_cycles, add_path_lengths, add_degree in variations:
                print(f'\nIteration {i}/{total}')
                nrmse_test, nrmse_range, nrmse_mean, nrmse_std, best_val_loss, train_time, total_epochs = \
                    evaluate_performance(data_dir=DATA_DIR,
                                model_class=model_class,
                                training_grid_codes=train_grids,
                                testing_grid_codes=[target],
                                batch_size=batch_size,
                                epochs=epochs,
                                shuffle=True,
                                add_cycles=add_cycles,
                                add_path_lengths=add_path_lengths,
                                add_degree=add_degree,
                                log_dir=log_dir,
                                plot=plot,
                                save_model=save_model,
                                eval_only=eval_only,
                                load_model_dir=load_model_dir,
                                experiment_id=i)
                results.append(
                    (
                        target,
                        add_cycles,
                        add_path_lengths,
                        add_degree,
                        nrmse_test,
                        nrmse_range, nrmse_mean, nrmse_std,
                        best_val_loss,
                        train_time,
                        total_epochs
                    )
                )
                print(f'\tBest val loss: {best_val_loss}\n\tNRMSE: {nrmse_test}')
                i += 1
        results_df = pd.DataFrame(results, columns=column_names)

        if save_results and log_dir:
            results_file = os.path.join(log_dir, 'results_ood.csv')
            results_df.to_csv(results_file)
            print(f'\nSaved OOD results to: {results_file}')

    if args.dc_pf:
        column_names = [
            'testing_grid',
            'nrmse_test',
            'nrmse_range', 'nrmse_mean', 'nrmse_std'
        ]
        results = []
        print('\nEvaluating dc pf...')
        for testing_grid in tqdm(grids_to_compare):
            nrmse_dc_pf, nrmse_range, nrmse_mean, nrmse_std = evaluate_dc_pf(DATA_DIR, testing_grid)
            results.append(
                (
                    testing_grid,
                    nrmse_dc_pf,
                    nrmse_range, nrmse_mean, nrmse_std
                )
            )
        results_df = pd.DataFrame(results, columns=column_names)
        if save_results and log_dir:
            results_file = os.path.join(log_dir, 'results_ood_dc_pf.csv')
            results_df.to_csv(results_file)
            print(f'\nSaved DC PF results to: {results_file}')

    if args.mmd:
        # Compare MMDs between train and test sets, and add to results df
        print('Calculating MMDs...')
        test_cases = get_mmd_test_cases(grids_to_compare)
        column_names = [
            'testing_grid',
            'mmd_degree',
            'mmd_laplacian'
        ]
        results = []
        for train_grids, target in tqdm(test_cases):
            mmd_degree, mmd_laplacian = \
                evaluate_tl_mmd(data_dir=DATA_DIR,
                                training_grid_codes=train_grids,
                                testing_grid_codes=[target])

            results.append(
                (target, mmd_degree, mmd_laplacian)
            )

        results_df = pd.DataFrame(results, columns=column_names)
        if save_results and log_dir:
            results_file = os.path.join(log_dir, 'results_ood_mmd.csv')
            results_df.to_csv(results_file)
            print(f'\nSaved MMD results to: {results_file}')

    print("\nTraining complete")