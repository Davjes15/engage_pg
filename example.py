import argparse

from models import GCN, ARMA_GNN
from training_utils import create_log_dir
from transfer_learning_experiment import evaluate_performance

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default='gcn',
    )
    parser.add_argument(
        "--training",
        default='1-LV-rural1--0-no_sw',
    )
    parser.add_argument(
        "--testing",
        default='1-LV-rural3--0-no_sw',
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--cycles",
        action="store_true"
    )
    parser.add_argument(
        "--path_lengths",
        action="store_true"
    )
    parser.add_argument(
        "--log_training",
        action="store_true"
    )
    parser.add_argument(
        "--plot",
        action="store_true"
    )
    parser.add_argument(
        "--save_model",
        action="store_true"
    )
    args = parser.parse_args()
    return args

DATA_DIR = 'outputs/2024-12-30_19:23:46/'

if __name__ == '__main__':
    model_classes = {'gcn': GCN, 'arma_gnn': ARMA_GNN}
    args = parse_args()
    log_dir = None
    if args.log_training:
        log_dir = create_log_dir()
    print()
    nrmse_test, best_val_loss, train_time, total_epochs = \
        evaluate_performance(data_dir=DATA_DIR,
                             model_class=model_classes[args.model],
                             training_grid_code=args.training,
                             testing_grid_code=args.testing,
                             epochs=args.epochs,
                             add_cycles=args.cycles,
                             add_path_lengths=args.path_lengths,
                             log_dir=log_dir,
                             plot=args.plot,
                             save_model=args.save_model)
    print()
    print('Test NRMSE:', nrmse_test)
    print('Best Validation Loss:', best_val_loss)
    print('Total Training Time (s)', train_time)
    print()