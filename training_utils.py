import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split
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

def setup_logging(log_dir):
    logfile = os.path.join(log_dir, 'log.txt')
    print(f'Logs will be saved in: {logfile}')
    sys.stdout = open(logfile, 'w')

def get_model_save_path(log_dir):
    return os.path.join(log_dir, 'model_weights')

def setup_pytorch_and_get_device():
    torch.manual_seed(12)

    device = (
        "cuda"
        if torch.cuda.is_available()
        # else "mps"
        # if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Training using {device}", flush=True)
    return device

def get_dataset(data_dir,
                grid_type,
                include_sources=False,
                cycles=False,
                path_lengths=False):
    pyg_dataset = get_pyg_graphs(data_dir, grid_type)
    if include_sources:
        pp_sources = get_pp_sources(data_dir, grid_type)
        for data, src in zip(pyg_dataset, pp_sources):
            data.src = src
    if cycles or path_lengths:
        pyg_dataset = add_augmented_features(pyg_dataset,
                                             cycles=cycles,
                                             path_lengths=path_lengths)
    return pyg_dataset

def get_dataloaders(data_dir,
                    training_grid='1-LV-rural1--0-no_sw',
                    testing_grid='1-LV-rural1--0-no_sw',
                    include_sources=False,
                    add_cycles=False,
                    add_path_lengths=False,
                    batch_size=16):
    loader_train = loader_val = loader_test = None
    if training_grid:
        train_dataset = get_dataset(data_dir,
                                    training_grid,
                                    include_sources=include_sources,
                                    cycles=add_cycles,
                                    path_lengths=add_path_lengths)
        train_split, val_split = random_split(train_dataset, TRAIN_VAL_SPLIT)

        loader_train = DataLoader(train_split,
                                batch_size=batch_size)
        loader_val = DataLoader(val_split,
                                batch_size=batch_size)
    if testing_grid:
        loader_test = DataLoader(get_dataset(data_dir,
                                            testing_grid,
                                            include_sources=include_sources,
                                            cycles=add_cycles,
                                            path_lengths=add_path_lengths),
                                batch_size=batch_size)
    return loader_train, loader_val, loader_test

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
    loss_fn = nn.MSELoss(reduction='mean') # Average over all elements

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

def plot_loss(model_classname, train_loss_vec, val_loss_vec, cycle=False, path_lengths=False):
    fig, ax = plt.subplots()
    ax.plot(train_loss_vec, label = 'train loss')
    ax.plot(val_loss_vec, label = 'val loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    title = f"{model_classname}, cycle: {cycle}, path_lengths: {path_lengths}"
    ax.set_title(title)
    plt.show(block=True)

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

def test_dc_opf(loader_test):
    # TODO: Add batching if too slow -> Can run pp, add them back to Data
    # object, and do the loss calculations in batches.
    nrmse_test = 0
    for data in loader_test.dataset:
        # Load the source network
        net = pp.from_json(data.src)
        # Run dc opf
        pp.rundcpp(net)
        # Put this in correct format to match the true data and get np array.
        np_pred_y = net.res_bus[['p_mw', 'q_mvar', 'vm_pu', 'va_degree']].values
        # Convert to tensor and replace nan (q_mwar) with 0.
        pred_y = torch.nan_to_num(torch.Tensor(np_pred_y), nan=0.0)
        loss = nrmse_loss(pred_y, data.y)
        nrmse_test += loss.item()
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