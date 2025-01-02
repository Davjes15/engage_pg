import os
import argparse

from models import GCN, ARMA_GNN
from training_utils import (
    create_log_dir,
    setup_logging,
    get_model_save_path,
    setup_pytorch_and_get_device,
    get_dataloaders,
    plot_loss,
    train,
    test
)

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

def evaluate(model_class,
             training_grid_code,
             testing_grid_code,
             epochs=1000,
             add_cycles=False,
             add_path_lengths=False,
             log_training=False,
             plot=False,
             save_model=False):
    # Log information about training run
    learning_rate=1e-3
    early_stopping=True
    patience=100
    best_val_weights=True
    print(locals(), flush=True)

    # Create artefacts for logging
    log_dir = ''
    if log_training or save_model:
        log_dir = create_log_dir()

    if log_training:
        setup_logging(log_dir)

    model_weights_path = ''
    if save_model:
        model_weights_path = get_model_save_path(log_dir)

    # PyTorch setup
    device = setup_pytorch_and_get_device()

    # Get data loaders
    data_dir = 'outputs/2024-12-30_19:23:46/'
    loader_train, loader_val, loader_test = get_dataloaders(
        data_dir=data_dir,
        training_grid=training_grid_code,
        testing_grid=testing_grid_code,
        add_cycles=add_cycles,
        add_path_lengths=add_path_lengths)

    # Create model
    input_dim = next(iter(loader_train)).x.shape[1]
    model = model_class(input_dim=input_dim).to(device)

    # Train the model
    train_loss_vec, val_loss_vec, train_time, best_val_loss = \
        train(model=model,
              device=device,
              loader_train=loader_train,
              loader_val=loader_val,
              epochs=epochs,
              learning_rate=learning_rate,
              early_stopping=early_stopping,
              patience=patience,
              best_val_weights=best_val_weights,
              save_model_to=model_weights_path)
    
    # Plot the model
    if plot:
        plot_loss(model_class.__name__,
                  train_loss_vec,
                  val_loss_vec,
                  add_cycles,
                  add_path_lengths)

    # Test the model
    nrmse_loss = test(model=model,
                      device=device,
                      loader_test=loader_test)
    
    return train_time, best_val_loss, nrmse_loss

if __name__ == '__main__':
    model_classes = {'gcn': GCN, 'arma_gnn': ARMA_GNN}
    args = parse_args()
    train_time, best_val_loss, nrmse_test = \
        evaluate(model_class=model_classes[args.model],
                 training_grid_code=args.training,
                 testing_grid_code=args.testing,
                 epochs=args.epochs,
                 add_cycles=args.cycles,
                 add_path_lengths=args.path_lengths,
                 log_training=args.log_training,
                 plot=args.plot,
                 save_model=args.save_model)