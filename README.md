# GERN: Graph Effective Resistance Networks

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14629731.svg)](https://doi.org/10.5281/zenodo.14629731)


GERN is a framework designed to improve the training of Graph Neural Networks (GNNs) by addressing common issues such as over-squashing and over-smoothing. GERN leverages the concept of effective resistance to generate Random Spanning Trees (RSTs) and Random Path Graphs (RPGs), enabling more efficient and scalable GNN training, particularly in small training set regimes.

## Table of Contents

- [GERN: Graph Effective Resistance Networks](#gern-graph-effective-resistance-networks)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Python Requirements](#python-requirements)
  - [Datasets](#datasets)
    - [Running Experiments](#running-experiments)
      - [Basic Usage](#basic-usage)
      - [Parameters](#parameters)
      - [Example Commands](#example-commands)
  - [Citation](#citation)
  - [License](#license)

## Installation

To get started with GERN, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/username/gern.git
   cd gern
   ```

## Python Requirements

The following Python libraries are required to run GERN:

- **[PyTorch Geometric (pyg)](https://pytorch-geometric.readthedocs.io/):** The main library used for graph-based deep learning. It requires PyTorch, which will be installed as a dependency.
- **Jupyter:** To run and interact with notebooks.
- **Matplotlib:** For creating plots and visualizations.
- **NumPy, Pandas, SciPy, Scikit-learn**
- **TQDM:** For displaying progress bars when running experiments.
- **[Open Graph Benchmark (ogb)](https://ogb.stanford.edu/):** For loading and working with larger graph datasets like OGBN-arXiv and OGBN-Products.
  
**Set up a Conda virtual environment (optional but recommended):**

To install the environment and use the Jupyter notebooks, run (assuming you already have `conda` and `ipykernel` installed):

    conda env create -f environment.yml
    python -m ipykernel install --user --name=gern

To update the environment:

    conda env update -f environment.yml --prune

This command installs also the required dependencies.

## Datasets

GERN supports several commonly used graph datasets, including:

- **Cora**
- **PubMed**
- **OGBN-arXiv**
- **AMiner-CS**
- **OGBN-Products**

To run the experiments, download the datasets from their respective sources, or use the provided dataset loading scripts in the repository. Ensure the datasets are placed in the appropriate `data/` directory.

### Running Experiments

To run experiments with GERN, the main script is located in `src/main.py`. Below is the usage and explanation of the various parameters you can configure.

#### Basic Usage

Navigate to the project directory and execute the main script:

```bash
python src/main.py -d cora -e rpg --split percentage --train_size 0.01
```

#### Parameters

- **-h, --help**: Show the help message and exit.
- **-r, --nr_runs**: Number of runs for the experiment (default: 1).
- **--start_seed**: Seed for random number generation to ensure reproducibility (default: 0).
- **-d, --dataset**: Specify the dataset to use. Options include:
  - `cora`
  - `pubmed`
  - `ogbn-arxiv`
  - `aminer`
  - `ogbn-products`
- **-e, --experiment**: Choose the experiment type. Options include:
  - `full_graph`
  - `rst`
  - `rpg`
  - `graph_saint`
  - `graph_sage`
  - `gern`
- **--num_layers**: Number of GNN layers (default: 2).
- **--hidden_channels**: Number of hidden channels in the GNN layers (default: 64).
- **--dropout**: Dropout rate to use during training (default: 0.5).
- **--lr**: Learning rate for the optimizer (default: 0.01).
- **--wd**: Weight decay (L2 regularization) (default: 5e-4).
- **--nr_rsts**: Number of Random Spanning Trees (RSTs) to use (default: 1).
- **--out**: Directory to save output results.
- **--use_gcn**: Use the GCN architecture if set (default option).
- **--use_sage**: Use the GraphSAGE architecture if set.
- **--use_bn**: Use Batch Normalization if set.
- **--epochs**: Maximum number of training epochs (default: 200).
- **--min_epochs**: Minimum number of training epochs (default: 10).
- **--val_step**: Frequency of validation steps during training.
- **--patience**: Number of validation steps to wait before early stopping (default: 10).
- **--device**: Device to use for computation (`cpu` or `cuda`).
- **--verbose**: Enable verbose output if set.
- **--batch_inference**: Use batch inference to speed up the process.
- **--split**: Specify the data split method. Options include:
  - `planetoid`
  - `percentage`
  - `public`
- **--num_train_per_class**: Number of training samples per class.
- **--train_size**: Proportion of the dataset to use for training, specified as a float (e.g., `0.05` for 5%).

#### Example Commands

**Run GERN on the Cora dataset with default settings:**

   ```bash
   python src/main.py -d cora -e rpg
   ```

**Run GERN on the Cora dataset with 20 nodes per class:**

   ```bash
   python src/main.py -d cora -e rpg --nr_rsts 250 --split planetoid --num_train_per_class 20 --use_gcn --wd 5e-4 --start_seed 0 --num_layers 2 --hidden_channels 64 --nr_runs 100 --start_seed 0
   ```

**Run an experiment with a small training set (1% of the dataset) on OGBN-Products:**

   ```bash
   python src/main.py -d ogbn-products -e rpg --split percentage --train_size 0.01
   ```

All experiment results, including performance metrics, will be saved in the specified output directory or the default location (`data/results`) if not specified.

For more detailed information on each parameter and its usage, refer to the help command:

```bash
python src/main.py -h
```

## Citation

If you use GERN in your research, please cite the following paper:

> Francesco Bonchi, Claudio Gentile, Francesco Paolo Nerini, André Panisson, and Fabio Vitale. 2025. Fast and Effective GNN Training through Sequences of Random Path Graphs. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1 (KDD '25), August 3–7, 2025, Toronto, ON, Canada. ACM, New York, NY, USA, 12 pages.

ArXiv extended paper:

> Bonchi, Francesco, Claudio Gentile, Francesco Paolo Nerini, André Panisson, and Fabio Vitale. Fast and Effective GNN Training with Linearized Random Spanning Trees. arXiv preprint arXiv:2306.04828. https://arxiv.org/abs/2306.04828 

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

For any questions or issues, please open an issue on the repository or contact the authors directly.