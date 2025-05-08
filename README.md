# ENGAGE: Evaluating Network Generalization for AC Grid Estimation

## Dependencies

### Updating Git Submodules

If you did not clone with the submodules (i.e. using something like `git clone --recurse-submodules https://gitlab.lrz.de/<PATH>/<TO>/<REPO>.git`), then you will need to run the following to initialize, fetch and checkout any nested submodules:

```
git submodule update --init --recursive
```

Additionally we have to make some edits to the submodules. All updates are documentated in the `submodule_updates/` directory, with the same path as the ones you should update. The current commands to update these submodules are files we have to update currently are:
```
cp submodule_updates/ggme/src/metrics/mmd.py ggme/src/metrics/mmd.py
cp submodule_updates/powerdata-gen/powerdata_gen/powergrid/topology.py powerdata-gen/powerdata_gen/powergrid/topology.py
```

From experience, it seems like there is also an error in part of the networkx code that can sometimes be triggered and should be fixed. We use miniconda so the instructions are targeted towards that environment framework, so you may need to adapt if necessary. For miniconda, navigate to your home directory (or where miniconda is located), and open the file: `miniconda3/envs/engage/lib/python3.10/site-packages/networkx/linalg/laplacianmatrix.py`. Here you need to change `sp.errstate` -> `np.errstate`.

### Environment Setup

Now that you have all the code, you will need to create and activate the new environment. Run:
```
conda env create -f environment.yaml
conda activate engage
```

## Getting Started

### Data Generation

To generate a dataset, you can use `graph_gen.py`. Running this with `-h` can show you all the options. To test if it works, without spending too much time, you can also run:

```
python graph_gen.py --dry_run
```

Now let us generate a tiny dataset for distribution grids for the "tomorrow" scenario from [SimBench](https://simbench.de/).

```
python graph_gen.py --size 5 --scenario 1
```

This will generate a training dataset of 5 grids for each reference distribution grid in the "tomorrow" scenario. It will also print some messages telling you where the files have been stored. You will want to remember the base data directory, under which the grid names are stored. For example, if you receive print messages such as `"Saving dataset in outputs/2025-01-15_10:58:53/1-MV-urban--1-no_sw/train/dataset.pt... completed"`, then your base data directory would be `outputs/2025-01-15_10:58:53/`.

#### Data Attributes

Each Data object in the dataset list has the following attributes and dimensions:

- *x*:
  - Dimension: (N, 7)
  - N is the number of nodes/buses in the network
  - There are 7 node features: [Slack?, PV?, PQ?, p_mw, q_mvar, vm_pu, va_degree]
- *edge_index*:
  - Dimension: (2, 2E)
  - E is the number of lines in the network
  - 2 because each the edge_index list assumes directed edges
- *edge_attr*:
  - Dimension: (2E, 5)
  - E is the number of lines in the network
  - There are 5 edge features: [trafo?, r_pu, x_pu, length, sc_voltage]
- *y*:
  - Dimension: (N, 4)
  - N is the number of nodes/buses in the network
  - There are 4 true node attributes: [p_mw, q_mvar, vm_pu, va_degree]
- *dc_pf*:
  - Dimension: (N, 4)
  - N is the number of nodes/buses in the network
  - There are 4 DC-PF approximated node attributes: [p_mw, q_mvar, vm_pu, va_degree]

### Running Experiments

After we have a dataset, we can run the `Cross-Context` and `Out-of-Distribution` experiments using the appropriate data location, or `data_dir`, that you noted from earlier. You can run each script using the `-h` command to see the list of potential arguments.

```
python cross_context_experiment.py -h
```
```
python out_of_distribution_experiment.py -h
```

For example, we can train the sample gcn model using the dataset you generated above. Assuming the `data_dir` for your self-generated data is `outputs/2025-01-15_10:58:53/`, we can use the following:
```
python cross_context_experiment.py --data_dir outputs/2025-01-15_10:58:53/ --model gcn --epochs 1 --save_results --scenario 1
```

### Benchmarking a New Model

To benchmark a new model, you will have to:

1. Create a new model in `models.py`, following the structure of the existing examples (including the constructor parameters)
2. Import the model in the appropriate experiment file (eg. in `cross_context_experiment.py`)
3. Add a new key for the model in the `model_classes` dictionary object of the experiment file
4. Re-run the experiment script using the key name as the model argument (eg. `--model <my_new_model>`)

## Reproduce Results from the Original Paper

The data directory is available at: TODO. Once downloaded, you can move/rename the directory however you like. For example, you can rename the data directory path to `data/engage_dataset/`. Using this data, you can run the experiments using the `cross_context_experiment.py` and `out_of_distribution_experiment.py` scripts.

To reproduce the results from our paper, you can use the following commands:

```
python cross_context_experiment.py --data_dir data/engage_dataset/ --model gcn --batch_size 64 --epochs 500 --dc_pf --mmd --save_results --scenario 1 --save_model
```

```
python out_of_distribution_experiment.py --data_dir data/engage_dataset/ --model gcn --batch_size 256 --epochs 500 --dc_pf --mmd --save_results --scenario 1 --save_model
```

These scripts will take a while to run, and will produce and print result files for you. For example, for the OOD experiment, you will see something like `out/.../results_ood.csv`, `out/.../results_ood_dc_pf.csv`, and `out/.../results_ood_mmd.csv`.

***Note:*** If you want to skip the DC PF and MMD calculations (because they are independent of our PF models), you can remove the `--dc_pf` and `--mmd` flags, since the results are all already recorded in `dc_pf_data.csv`, `cc_mmds_data.csv`, and `ood_mmds_data.csv`.

The results of these output files are then all analyzed in `results_analysis.ipynb`, where the g_score is also calculated. Towards the top of the notebook, you will need to input the paths to your own result files.
