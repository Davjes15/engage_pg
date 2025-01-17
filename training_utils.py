import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import pandapower as pp
from tqdm import tqdm

import time
import os
import sys

from graph_utils import get_networkx_graph, get_pyg_graphs, get_pp_sources, add_augmented_features

# Load graph eval library
GRAPH_EVAL_PATH = os.path.abspath('ggme/src/')
sys.path.append(GRAPH_EVAL_PATH)
from evaluate import evaluate_mmd as ggme_evaluate_mmd
from correlation import compute_correlation
from metrics.kernels import gaussian_kernel
from metrics.descriptor_functions import degree_distribution, normalised_laplacian_spectrum
# Reset path
sys.path.pop()

TRAIN_VAL_SPLIT = [0.8, 0.2]

def create_log_dir(model_classname):
    log_dir = os.path.join('out',
                            model_classname,
                            time.strftime('%Y%m%d-%H%M%S'))
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def setup_file_output(log_dir):
    logfile = os.path.join(log_dir, 'log.txt')
    print(f'Logs will be saved in: {logfile}')
    sys.stdout = open(logfile, 'w')

def get_model_save_path(log_dir):
    return os.path.join(log_dir, 'model_weights')

def setup_pytorch():
    torch.manual_seed(12)
    return

def get_device():
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        # else "mps"
        # if torch.backends.mps.is_available()
        else "cpu"
    )
    return device

DATASET_CACHE = {}

def get_dataset(data_dir,
                grid_types,
                init_dc=False,
                cycles=False,
                path_lengths=False):
    complete_dataset = []
    for grid in grid_types:
        pyg_dataset = None
        id = ':'.join([str(v) for v in [grid, cycles, path_lengths]])
        if id in DATASET_CACHE:
            print('Cache hit:', id)
            pyg_dataset = DATASET_CACHE[id]
        else:
            print('Cache miss:', id)
            pyg_dataset = get_pyg_graphs(data_dir, grid)
            if cycles or path_lengths:
                pyg_dataset = add_augmented_features(pyg_dataset,
                                                    cycles=cycles,
                                                    path_lengths=path_lengths)
            DATASET_CACHE[id] = pyg_dataset
        complete_dataset.extend(pyg_dataset)

    if init_dc:
        for data in complete_dataset:
            data.x[:, 3:7] = data.dc_pf

    return complete_dataset

def get_dataloaders(data_dir,
                    training_grids=['1-LV-rural1--0-no_sw'],
                    testing_grids=['1-LV-rural1--0-no_sw'],
                    init_dc=False,
                    add_cycles=False,
                    add_path_lengths=False,
                    batch_size=16,
                    shuffle=False):
    loader_train = loader_val = loader_test = None
    if training_grids:
        train_dataset = get_dataset(data_dir,
                                    training_grids,
                                    init_dc=init_dc,
                                    cycles=add_cycles,
                                    path_lengths=add_path_lengths)
        train_split, val_split = random_split(train_dataset, TRAIN_VAL_SPLIT)

        loader_train = DataLoader(train_split,
                                batch_size=batch_size,
                                shuffle=shuffle)
        loader_val = DataLoader(val_split,
                                batch_size=batch_size,
                                shuffle=shuffle)
    if testing_grids:
        loader_test = DataLoader(get_dataset(data_dir,
                                            testing_grids,
                                            init_dc=init_dc,
                                            cycles=add_cycles,
                                            path_lengths=add_path_lengths),
                                batch_size=batch_size,
                                shuffle=shuffle)
    return loader_train, loader_val, loader_test

def weighted_mse_loss(pred, target, eps=1e-8):
    # To give equal importance to smaller and larger vectors, we weight the loss
    # by the inverse of the true vector’s norm.

    # Compute the L2 norm of the true vectors
    target_norm = torch.norm(target, dim=-1, keepdim=True) + eps  # Shape: (N, 1)
    # Weight for each vector is the inverse of its norm
    weights = 1.0 / target_norm  # Shape: (N, 1)
    # Compute the element-wise MSE
    mse = nn.functional.mse_loss(pred, target, reduction='none')  # Shape: (N, D)
    # Apply weights and compute the mean
    weighted_mse = weights * mse  # Broadcasting over (N, D)
    # Return the mean loss across all elements
    return weighted_mse.mean()

def train(model,
          device,
          loader_train,
          loader_val,
          epochs=100,
          learning_rate=1e-3,
          early_stopping=True,
          patience=100,
          best_val_weights=True,
          save_model_to='',
          log_epochs=False):

    # Configure hyperparameters
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # loss_fn = nn.MSELoss(reduction='mean') # Average over all elements
    loss_fn = weighted_mse_loss

    # Variables to track best model
    best_val_loss = np.Inf
    best_weights = None

    # Setup arrays to track training performance
    train_loss_vec = np.empty(epochs)
    train_loss_vec[:] = np.nan 
    val_loss_vec = np.empty(epochs)
    val_loss_vec[:] = np.nan

    # Run timed train-eval loop
    start = time.time()
    total_epochs = -1
    for epoch in tqdm(range(epochs)):
        # Train
        model.train()
        loss_train = 0
        for batch_train in loader_train:
            batch_train = batch_train.to(device)
            optimizer.zero_grad()
            pred = model(batch_train)
            loss = loss_fn(pred, batch_train.y)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()*batch_train.num_graphs
        loss_train /= len(loader_train.dataset)

        # Validate
        model.eval()
        loss_val = 0
        for batch_val in loader_val:
            batch_val = batch_val.to(device)
            pred = model(batch_val)
            loss = loss_fn(pred, batch_val.y)
            loss_val += loss.item()*batch_val.num_graphs
        loss_val /= len(loader_val.dataset)

        # Early stopping and update of best model
        if early_stopping == True or best_val_weights == True:
            if loss_val < best_val_loss:
                wait = 0
                best_weights = model.state_dict()
                best_val_loss = loss_val
            elif wait >= patience and early_stopping == True:
                total_epochs = epoch
                break
            else:
                wait += 1

        # Track model performance
        train_loss_vec[epoch] = loss_train
        val_loss_vec[epoch] = loss_val
        if log_epochs and epoch % 100 == 99:
            print('Epoch: {} Train Loss: {:.6f} Valid Loss: {:.6f}'
                    .format(epoch + 1, loss_train, loss_val), flush=True)

    # Total training time
    train_time = time.time() - start

    # Total num epochs (if stopped early)
    total_epochs = epochs if total_epochs == -1 else total_epochs

    if best_val_weights == True:
        model.load_state_dict(best_weights)

    if save_model_to:
        torch.save(model.state_dict(), save_model_to)
        print(f'Model weights saved to: {save_model_to}')

    return train_loss_vec, val_loss_vec, best_val_loss, train_time, total_epochs

def plot_loss(log_dir,
              model_classname,
              train_loss_vec,
              val_loss_vec,
              cycle=False,
              path_lengths=False,
              fig_id=0):
    filename = os.path.join(log_dir, f'fig_{fig_id}.png')
    fig, ax = plt.subplots()
    ax.plot(train_loss_vec, label = 'train loss')
    ax.plot(val_loss_vec, label = 'val loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    title = f"{model_classname}, cycle: {cycle}, path_lengths: {path_lengths}"
    ax.set_title(title)
    plt.savefig(filename)
    print(f'Figure saved to: {filename}')

def nrmse_loss(y_pred, y_real):
    element_mse = torch.nn.MSELoss(reduction='none')(y_pred, y_real)
    sqrt_n_mean = torch.sqrt(torch.mean(element_mse, dim=0)) # sqrt(Mean of N samples)
    nrmse = torch.mean(sqrt_n_mean) # Normalized across features
    return nrmse

def test(model,
         device,
         loader_test):
    model.eval()
    nrmse_test = 0
    for batch_test in loader_test:
        batch_test = batch_test.to(device)
        pred = model(batch_test)
        loss = nrmse_loss(pred, batch_test.y)
        nrmse_test += loss.item()*batch_test.num_graphs
    nrmse_test /= len(loader_test.dataset)

    return nrmse_test

def test_dc_pf(device, loader_test):
    nrmse_test = 0
    for dc_batch in loader_test:
        dc_batch.to(device)
        loss = nrmse_loss(dc_batch.dc_pf, dc_batch.y)
        nrmse_test += loss.item()*dc_batch.num_graphs
    nrmse_test /= len(loader_test.dataset)

    return nrmse_test

def evaluate_mmd(training_dataset, testing_dataset):
    graphs_train = [get_networkx_graph(pyg_graph, include_features=False)
                    for pyg_graph in training_dataset]
    graphs_test = [get_networkx_graph(pyg_graph, include_features=False)
                   for pyg_graph in testing_dataset]

    # Degree Distrubtion
    sigma_degree = 1e3
    mmd_degree = ggme_evaluate_mmd(
        graphs_dist_1=graphs_train,
        graphs_dist_2=graphs_test,
        function=degree_distribution,
        kernel=gaussian_kernel,
        density=True,
        use_linear_approximation=False,
        sigma=sigma_degree)

    # Laplacian Spectrum
    sigma_laplacian = 1e-2
    mmd_laplacian = ggme_evaluate_mmd(
        graphs_dist_1=graphs_train,
        graphs_dist_2=graphs_test,
        function=normalised_laplacian_spectrum,
        kernel=gaussian_kernel,
        density=True,
        use_linear_approximation=False,
        sigma=sigma_laplacian)
    
    return mmd_degree, mmd_laplacian