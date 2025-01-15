# Graph PF

## Getting Started

If you did not clone with the submodules (i.e. using something like `git clone --recurse-submodules https://gitlab.lrz.de/<PATH>/<TO>/<REPO>.git`), then you will need to run the following to initialize, fetch and checkout any nested submodules:

```
git submodule update --init --recursive
```

Additionally we have to make some edits to the submodules. All updates are documentated in the `submodule_updates/` directory, with the same path as the ones you should update. The current commands to update these submodules are files we have to update currently are:
```
cp submodule_updates/ggme/src/metrics/mmd.py ggme/src/metrics/mmd.py
cp submodule_updates/powerdata-gen/powerdata_gen/powergrid/topology.py powerdata-gen/powerdata_gen/powergrid/topology.py
```

From experience, it seems like there is also an error in part of the networkx code that can sometimes be triggered and should be fixed. We use miniconda so the instructions are targeted towards that environment framework, so you may need to adapt if necessary. For miniconda, navigate to your home directory (or where miniconda is located), and open the file: `miniconda3/envs/graph-pf/lib/python3.10/site-packages/networkx/linalg/laplacianmatrix.py`. Here you need to change `sp.errstate` -> `np.errstate`.

Now that you have all the code, you will need to create and activate the new environment. Run:
```
conda env create -f environment.yaml
conda activate graph-pf
```

## Toy Example

In general, to generate a dataset, you can use `graph_gen.py`. Running this with `-h` can show you all the options. To test if it works, without spending too much time, you can also run:

```
python graph_gen.py --dry_run
```

Now let us generate a tiny dataset for distribution grids for the "tomorrow" scenario from simbench, and run the transfer learning experiment on it. To do this, you will first need to run:

```
python graph_gen.py --size 5 --scenario 1
```

This will generate a training dataset of 5 grids for each reference distribution grid in the "tomorrow" scenario. It will also print some messages telling you where the files have been stored. You will want to remember the base data directory, under which the grid names are stored. For example, if you receive print messages such as `"Saving dataset in outputs/2025-01-15_10:58:53/1-MV-urban--1-no_sw/train/dataset.pt... completed"`, then you base data directory would be `outputs/2025-01-15_10:58:53/`.

After this dataset is generated, the transfer learning experiment can be run using the `data_dir` you noted from earlier:
```
python transfer_learning_experiment.py --data_dir <data_dir> --model gcn --epochs 1 --save_results --scenario 1
```

