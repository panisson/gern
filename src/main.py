"""
Main script to run experiments.

usage: main.py [-h] [-r NR_RUNS] [-d {cora,pubmed,ogbn-arxiv}] -e
               {full_graph,rst,rst_lp,rst_lp2,rpg,rpg_lp,rpg_lp2,full_lp}
               [--nr_rsts NR_RSTS] [--lp_max_dist LP_MAX_DIST]
               [--dropout DROPOUT]

RST-GNN experiments.

options:
  -h, --help            show this help message and exit
  -r NR_RUNS, --nr_runs NR_RUNS
                        number of runs for each experiment
  -d {cora,pubmed,ogbn-arxiv}, --dataset {cora,pubmed,ogbn-arxiv}
                        which dataset to use
  -e {full_graph,rst,rst_lp,rst_lp2,rpg,rpg_lp,rpg_lp2,full_lp}
                        set the experiment(s)
  --nr_rsts NR_RSTS     the number of RSTs
  --lp_max_dist LP_MAX_DIST
                        the maximum distance from labeled node in label
                        propagation
  --dropout DROPOUT     the dropout rate

Author: André Panisson

Example to run all experiments with 10 runs and Cora dataset:
python main.py --nr_runs 10 \
    -e full_graph -e rst -e rst_lp -e rst_lp2 -e rpg -e rpg_lp -e rpg_lp2 -e full_lp \
    --dataset cora
"""

import argparse
import os
import os.path as osp
import utils

from data_loader import *
import experiments_pyg as exp
import rst_utils as rstu

# try:
#     # Load Torch Geometric if it is available
#     import torch_geometric as tg
#     from data_loader import load_cora, load_pubmed, load_ogbn_arxiv, load_ogbn_mag
#     import experiments_pyg as exp
# except:
#     # Otherwise, load tensorflow
#     from data_loader_tf import load_cora, load_pubmed, load_ogbn_arxiv, load_ogbn_mag
#     import experiments_tf as exp

exp_classes = {
    "full_graph": exp.FULL,
    "rst": exp.RST,
    "rpg": exp.RPG,
    "rpg_lp": exp.RPG_LP,
    "rst_lp": exp.RST_LP,
    "full_lp": exp.FULL_LP,
    "rst_lp2":  exp.RST_LP2,
    "rpg_lp2": exp.RPG_LP2,
    "graph_saint": exp.GRAPH_SAINT,
    "graph_sage": exp.SAGE_EXP,
    "random": exp.RANDOM,
    "gern": exp.GERN
}


#### Command line Arguments ####

parser = argparse.ArgumentParser(
    description="RST-GNN experiments.",
    epilog="Author: André Panisson"
)
parser.add_argument(
    "-r", "--nr_runs", type=int, default=10,
    help="number of runs for each experiment"
)
parser.add_argument('--start_seed', type=int, default=0)
parser.add_argument(
    "-d", "--dataset", default="cora",
    choices=["cora", "pubmed", "flickr", "yelp", "reddit2", 
             "ogbn-arxiv", "ogbn-mag", "aminer"],
    help="which dataset to use"
)
parser.add_argument(
    '-e', action='append', required=True, dest="experiment",
    choices=list(exp_classes.keys()),
    help='set the experiment(s)'
)
parser.add_argument("--nr_rsts", type=int, default=250,
    help="the number of RSTs"
)
parser.add_argument(
    "--lp_max_dist", type=int, default=20,
    help="the maximum distance from labeled node in label propagation"
)
parser.add_argument("--out", default="../data/results",
    help="the output path for results"
)
parser.add_argument(
    "--nr_shuffled_features", type=int, default=0,
    help="the number of nodes with shuffled features for testing robustness"
)
# Model hyperparameters
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--wd', type=float, default=0.0)
parser.add_argument('--use_sage', action='store_true')
parser.add_argument('--use_gcn', action='store_true')
parser.add_argument('--use_bn', action='store_true')

# Training hyperparameters
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--min_epochs', type=int, default=100)
parser.add_argument('--val_step', type=int, default=1)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--device', type=int, default=0)

parser.add_argument('--verbose', type=int, default=0)

# Batch or not
parser.add_argument('--batch_inference', action='store_true')

# Train test split
parser.add_argument('--split', type=str, default="planetoid",
    dest="split_type", choices=["planetoid", "percentage", "public"])
parser.add_argument('--num_train_per_class', type=int, nargs='+',
    default=[5,10,20], dest="num_train_per_class_list",
    help="the number of train nodes per class (planetoid split)"
)
parser.add_argument('--train_size', type=float, default=0.8,
    help="the train size (percentage split)"
)

args = parser.parse_args()
nr_runs = args.nr_runs
experiments = args.experiment
dataset = args.dataset
lp_max_dist = args.lp_max_dist
nr_rsts = args.nr_rsts
dropout = args.dropout
num_train_per_class_list = args.num_train_per_class_list
results_path = osp.join(osp.dirname(osp.realpath(__file__)), args.out)
nr_shuffled_features = args.nr_shuffled_features
split_type = args.split_type

device_nr = args.device
exp.set_device(device_nr)

vargs = vars(args)
vargs.pop("experiment")

# print(vargs)

print(f"Saving results to path '{osp.abspath(results_path)}'")
os.makedirs(results_path, exist_ok=True)

print(f"Loading {dataset} dataset.")
if dataset == "cora":
    data = load_cora()
elif dataset == "pubmed":
    data = load_pubmed()
elif dataset == "flickr":
    data = load_flickr()
elif dataset == "yelp":
    data = load_yelp()
elif dataset == "reddit2":
    data = load_reddit2()
elif dataset == "ogbn-arxiv":
    data = load_ogbn_arxiv()
elif dataset == "ogbn-mag":
    data = load_ogbn_mag()
elif dataset == "aminer":
    data = load_aminer()
else:
    raise Exception("Unknown dataset")
print(f"Loaded, {data.num_nodes} nodes, {data.num_edges} edges.") # type: ignore

if nr_shuffled_features > 0:
    data = exp.poison_data(data, nr_shuffled_features)

# check if dataset has sufficient number of samples for training
for num_train_per_class in num_train_per_class_list:
    exp.verify_train_size(data, num_train_per_class)

if nr_rsts > 0:
    print(f"Building {nr_rsts} RSTs")
    if dataset in ["yelp", "reddit2", "ogbn-arxiv", "ogbn-mag", "aminer"]:
        rstu.build_rsts_parallel(data, nr_rsts)
    else:
        rstu.build_rsts_parallel(data, nr_rsts, cached=False)

def run_experiments(experiment, split):
    experimental_data = []
    experiment_runner = exp.ExperimentRunner(exp_class, data=data, 
                                             split=split, **vargs)
    for results in experiment_runner.run_many_experiments(
            data, experiment, nr_experiments=nr_runs,
            num_train_per_class=num_train_per_class):
        results.update({'method': experiment,
                        'num_train_per_class': num_train_per_class})
        experimental_data.append(results)

    utils.write_report(experimental_data=experimental_data,
            results_path=results_path, experiment=experiment,
            num_train_per_class=num_train_per_class, **vargs)

for experiment in experiments:
    print(f"Running experiment: {experiment}")
    exp_class = exp_classes.get(experiment, exp.Experiment)
    if split_type == "planetoid":
        for num_train_per_class in num_train_per_class_list:
            split = utils.PlanetoidSplit(data.num_classes,  # type: ignore
                                         num_train_per_class)
            run_experiments(experiment, split)
    elif split_type == "percentage":
        split = utils.TrainTestSplit(train_size=args.train_size)
        run_experiments(experiment, split)

    elif split_type == "public":
        run_experiments(experiment, None)
    
    else:
        raise NotImplementedError()

