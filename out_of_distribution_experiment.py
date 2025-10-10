import os

import pandas as pd
import numpy as np
from tqdm import tqdm

from models import GCN, ARMA_GNN
from training_utils import (
    create_log_dir,
    setup_pytorch,
    get_generalization_score
)
from graph_utils import get_dist_grid_codes
from cross_context_experiment import (
    parse_args,
    evaluate_performance,
    evaluate_dc_pf,
    evaluate_cc_mmd
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
            'nrmse_test'
        ]
        results = []
        print('\nEvaluating dc pf...')
        for testing_grid in tqdm(grids_to_compare):
            nrmse_dc_pf = evaluate_dc_pf(DATA_DIR, testing_grid)
            results.append(
                (
                    testing_grid,
                    nrmse_dc_pf,
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
                evaluate_cc_mmd(data_dir=DATA_DIR,
                                training_grid_codes=train_grids,
                                testing_grid_codes=[target],
                                sigma_degree=1e2,
                                sigma_laplacian=1e-2)

            results.append(
                (target, mmd_degree, mmd_laplacian)
            )

        results_df = pd.DataFrame(results, columns=column_names)
        if save_results and log_dir:
            results_file = os.path.join(log_dir, 'results_ood_mmd.csv')
            results_df.to_csv(results_file)
            print(f'\nSaved MMD results to: {results_file}')

    if args.mmd and not args.skip_experiment:
        # Correlate MMDs with performance metrics
        print('Correlating MMDs with performance metrics to get s score...')
        perf_file = os.path.join(log_dir, 'results_ood.csv')
        mmd_file = os.path.join(log_dir, 'results_ood_mmd.csv')
        perf_df = pd.read_csv(perf_file, index_col=0)
        mmd_df = pd.read_csv(mmd_file, index_col=0)

        gen_stats = pd.DataFrame(columns=['model',
                                          'mean_nrmse',
                                          'std_nrmse',
                                          'mmd_range_degree',
                                          'mmd_range_laplacian',
                                          'g_score_degree',
                                          'g_score_laplacian'])

        if args.dc_pf:
            dc_pf_file = os.path.join(log_dir, 'results_ood_dc_pf.csv')
            dc_pf_df = pd.read_csv(dc_pf_file, index_col=0)
            mean_nrmse, std_nrmse, mmd_range_degree, g_score_degree = get_generalization_score(np.repeat(0, len(dc_pf_df)), dc_pf_df['nrmse_test'].to_numpy())
            _, _, mmd_range_laplacian, g_score_laplacian = get_generalization_score(np.repeat(0, len(dc_pf_df)), dc_pf_df['nrmse_test'].to_numpy())

            gen_stats.loc[len(gen_stats)] = [
                'dc_pf',
                mean_nrmse,
                std_nrmse,
                mmd_range_degree,
                mmd_range_laplacian,
                g_score_degree,
                g_score_laplacian
            ]

        models_variations = {
            # (cycles, path_lengths, degree)
            'ref': (False, False, False),
            'degree': (False, False, True),
            'cycles': (True, False, False),
            'paths': (False, True, False),
        }

        def get_model_variation_df(performance_df, cycles, path_lengths, degree):
            df = performance_df
            data_df = df[(df['cycles'] == cycles) & (df['path_lengths'] == path_lengths) & (df['degree'] == degree)]
            return data_df

        # Calculate generalization statistics for each model variation
        for name, config in models_variations.items():
            model_df = get_model_variation_df(perf_df, *config)
            model_df = model_df.merge(mmd_df,
                                  left_on=['testing_grid'],
                                  right_on=['testing_grid'],
                                  how='left')
            mean_nrmse, std_nrmse, mmd_range_degree, g_score_degree = get_generalization_score(model_df['mmd_degree'].to_numpy(), model_df['nrmse_test'].to_numpy())
            _, _, mmd_range_laplacian, g_score_laplacian = get_generalization_score(model_df['mmd_laplacian'].to_numpy(), model_df['nrmse_test'].to_numpy())

            gen_stats.loc[len(gen_stats)] = [
                name,
                mean_nrmse,
                std_nrmse,
                mmd_range_degree,
                mmd_range_laplacian,
                g_score_degree,
                g_score_laplacian
            ]

        if save_results and log_dir:
            results_file = os.path.join(log_dir, 'results_ood_gen_stats.csv')
            gen_stats.to_csv(results_file)
            print(f'\nSaved generalization statistics to: {results_file}')

    print("\nTraining complete")