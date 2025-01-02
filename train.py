import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt

from collections import OrderedDict
import time
import os
import sys

from models import GCN, ARMA_GNN
from graph_utils import get_pyg_graphs, add_augmented_features

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--training",
    default='1-LV-rural1--0-no_sw',
)
parser.add_argument(
    "--testing",
    default='1-LV-rural3--0-no_sw',
)
parser.add_argument(
    "--preprocess",
    action="store_true"
)
parser.add_argument(
    "--cycles",
    action="store_true"
)
parser.add_argument(
    "--paths",
    action="store_true"
)
args = parser.parse_args()

torch.manual_seed(12)

# Tunables and other options
P = OrderedDict([
    ('model', 'gcn'), # Model: 'gcn', 'arma_gnn'
    ('output_base_dir', 'outputs/2024-12-30_19:23:46/'),
    ('training_grid', args.training),
    ('testing_grid', args.testing), 
    ('preprocess', args.preprocess),
    ('epochs', 2000), # Number of epochs
    ('learning_rate', 1e-3),
    ('batch_size', 16),
    ('early_stopping', True),
    ('patience', 500), # Patience for early stopping
    ('best_val_weights', True), # Return best validation weights
    ('plot_loss', True),
    ('save_model', False),
    ('log_training', False),
    ('test_model', True)
])

if P['log_training'] or P['save_model']:
    SAVE_PATH = os.path.join('out',
                             P['model'],
                             time.strftime('%Y%m%d-%H%M%S'))
    os.makedirs(SAVE_PATH, exist_ok=True)

if P['log_training']:
    logfile = os.path.join(SAVE_PATH, 'log.txt')
    print('\nLogs will be saved in: ' + logfile)
    sys.stdout = open(logfile, 'w')

model_classes = {'gcn': GCN, 'arma_gnn': ARMA_GNN}

##########################################################################
# PYTORCH SETUP
##########################################################################

device = (
    "cuda"
    if torch.cuda.is_available()
    # else "mps"
    # if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Training using {device}", flush=True)
print(P, flush=True)

##########################################################################
# LOAD TRAINING AND VALIDATION DATASETS
##########################################################################

def preprocess(pyg_dataset):
    return pyg_dataset

def get_dataset(output_base_dir, grid_type, split='train', preprocess=False):
    pyg_dataset = get_pyg_graphs(output_base_dir, grid_type, split)
    if preprocess:
        pyg_dataset = add_augmented_features(pyg_dataset,
                                             cycles=args.cycles,
                                             path_lengths=args.paths)
    return pyg_dataset

loader_train = DataLoader(get_dataset(P['output_base_dir'],
                                         P['training_grid'],
                                         split='train',
                                         preprocess=P['preprocess']),
                          batch_size=P['batch_size'])
loader_val = DataLoader(get_dataset(P['output_base_dir'],
                                       P['training_grid'],
                                       split='val',
                                       preprocess=P['preprocess']),
                        batch_size=P['batch_size'])
loader_test = DataLoader(get_dataset(P['output_base_dir'],
                                        P['testing_grid'],
                                        split='train',
                                        preprocess=P['preprocess']),
                         batch_size=P['batch_size'])

##########################################################################
# MODEL SETUP AND TRAINING
##########################################################################
input_dim = next(iter(loader_train)).x.shape[1]
model = model_classes[P['model']](input_dim=input_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

restore_best_weights = P['best_val_weights']
early_stopping = P['early_stopping']
patience = P['patience']

best_val_loss = np.Inf
best_weights = None

epochs = P['epochs']
train_loss_vec = np.empty(epochs)
train_loss_vec[:] = np.nan 
val_loss_vec = np.empty(epochs)
val_loss_vec[:] = np.nan
start = time.time()
for epoch in range(epochs):
    model.train()
    loss_train = 0
    for batch_train in loader_train:
        batch_train = batch_train.to(device)
        optimizer.zero_grad()
        pred = model(batch_train)
        loss = loss_fn(pred, batch_train.y) # Average loss over the batch items.
        loss.backward()
        optimizer.step()
        loss_train += loss.item()*batch_train.num_graphs
    loss_train /= len(loader_train.dataset)

    model.eval()
    loss_val = 0
    for batch_val in loader_val:
        batch_val = batch_val.to(device)
        pred = model(batch_val)
        loss = loss_fn(pred, batch_val.y) # Average loss over the batch items.
        loss_val += loss.item()*batch_val.num_graphs
    loss_val /= len(loader_val.dataset)

    if early_stopping == True or restore_best_weights == True:
        if loss_val < best_val_loss:
            wait = 0
            best_weights = model.state_dict()
            best_val_loss = loss_val
        elif wait >= patience and early_stopping == True:
            break
        else:
            wait += 1

    train_loss_vec[epoch] = loss_train
    val_loss_vec[epoch] = loss_val
    if epoch % 10 == 9:
        print('Epoch: {} Train Loss: {:.6f} Valid Loss: {:.6f}'
                .format(epoch + 1, loss_train, loss_val), flush=True)

train_time = time.time() - start
print(f"Total Training Time (s): {train_time}", flush=True)

if restore_best_weights == True:
    print('Best Validation Loss:', best_val_loss)
    model.load_state_dict(best_weights)

if P['save_model'] == True:
    torch.save(model.state_dict(), SAVE_PATH + '/model_weights')

##########################################################################
# PLOT LOSS (OPTIONAL)
##########################################################################

if P['plot_loss'] == True:
    fig, ax = plt.subplots()
    ax.plot(train_loss_vec, label = 'train loss')
    ax.plot(val_loss_vec, label = 'val loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.set_title(f"{P['model']} {'with' if P['preprocess'] else 'without'} augmented features")
    plt.show(block=True)

##########################################################################
# TESTING (OPTIONAL)
##########################################################################

def nrmse_loss(y_pred, y_real):
    element_mse = torch.nn.MSELoss(reduction='none')(y_pred, y_real)
    sqrt_n_mean = torch.sqrt(torch.mean(element_mse, dim=0)) # sqrt(Mean of N samples)
    nrmse = torch.mean(sqrt_n_mean) # Normalized across features
    return nrmse

if P['test_model']:
    nrmse_loss_fn = lambda pred, real: torch.sqrt(nn.MSELoss()(pred, real)) / torch.var(real, dim=1, keepdim=False)
    model.eval()
    loss_test = 0
    for batch_test in loader_test:
        batch_test = batch_test.to(device)
        pred = model(batch_test)
        loss = nrmse_loss(pred, batch_test.y) # Average loss over the batch items.
        loss_test += loss.item()*batch_test.num_graphs
    loss_test /= len(loader_test.dataset)

    nrmse_val = loss_test
    print(f"Test NRMSE: {nrmse_val}", flush=True)
