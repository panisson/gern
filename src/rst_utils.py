import os
import gzip
import pickle
import time

import numpy as np
import torch
import torch_geometric as tg
import tqdm
import utils
import rst
from scipy import sparse

def edge_index_from_rpg(rpg, device):
    return torch.tensor(
        np.vstack([np.hstack([rpg[:-1],rpg[1:]]), 
                   np.hstack([rpg[1:],rpg[:-1]])])).to(device)

# Random Spanning Tree generation functions

class RST_Repository():
    
    instance = None
    
    def __init__(self):
        self.rsts = []
        self.rpgs = []
        self.edge_index_list = {}
    
    @classmethod
    def get_instance(cls):
        if cls.instance is None:
            cls.instance = RST_Repository()
        return cls.instance
    
    def to(self, device):
        self.device = device
    
    def reset(self):
        self.rsts = []
        self.rpgs = []
        self.edge_index_list = {}
    
    def add_rst(self, rst):
        self.rsts.append(rst)
        
    def add_rpg(self, rpg):
        self.rpgs.append(rpg)
    
    def get_edge_index(self, rpg_index):
        if rpg_index not in self.edge_index_list:
            rpg = self.rpgs[rpg_index]
            self.edge_index_list[rpg_index] = edge_index_from_rpg(rpg, self.device)
            
        return self.edge_index_list[rpg_index]
    
    def save(self, path=None):
        if path is None:
            absolute_path = os.path.dirname(__file__)
            relative_path = "../data/arxiv/rst_arxiv.pickle"
            path = os.path.join(absolute_path, relative_path)
        with gzip.open(path, 'wb') as f:
            pickle.dump([self.rsts, self.rpgs], f)
    
    def load(self, path=None):
        self.reset()
        if path is None:
            absolute_path = os.path.dirname(__file__)
            relative_path = "../data/arxiv/rst_arxiv.pickle"
            path = os.path.join(absolute_path, relative_path)
        with gzip.open(path, 'rb') as f:
            self.rsts, self.rpgs = pickle.load(f)
    
rst_repo = RST_Repository.get_instance()

# We explicitly define this variable as global for multiprocessing performance.
# From the docs: On Unix a child process can make use of a shared resource 
#   created in a parent process using a global resource.
global Gg

def build_rst(seed):
    global csgraph
    tree = rst.mixed_random_spanning_tree(csgraph, seed)
    cstree = sparse.csr_matrix(
        (np.ones(tree.shape[1], dtype=np.uint8), (tree[0], tree[1])),
        shape=csgraph.shape)
    cstree = cstree + cstree.T

    tg.seed_everything(seed)
    utils.set_seed(seed)
    rpg = np.array(utils.sp_depth_first_search(
        cstree.shape[0], cstree.indptr, cstree.indices))
    return tree, rpg

class BuildRstCached():

    def __init__(self, replace=False, path="") -> None:
        self.replace = replace
        self.path = path

    def __call__(self, seed):
        
        file_path =  f"{self.path}/rst_{seed}.pickle"
        if os.path.exists(file_path) and not self.replace:
            with gzip.open(file_path, 'rb') as f:
                rst, rpg, seed, duration = pickle.load(f)
            return rst, rpg
    
        global Gg
        before = time.time()
        rst, rpg = build_rst(seed)
        duration = time.time()-before
        with gzip.open(file_path, 'wb') as f:
            pickle.dump((rst, rpg, seed, duration), f)
        return rst, rpg
    
def build_rsts(data, nr_rsts):
    global csgraph
    csgraph = tg.utils.to_scipy_sparse_matrix(data.edge_index).tocsr()
    for seed in tqdm.tqdm(range(nr_rsts)):
        rst, rpg = build_rst(seed)
        rst_repo.add_rst(rst)
        rst_repo.add_rpg(rpg)

def build_rsts_parallel(data, nr_rsts, cached=True):
    
    # Use global graph to avoid passing as parameter to parallel funcitons
    # global Gg
    # Gg = utils.Graph.from_edge_index(data.edge_index.cpu().detach().numpy())
    
    global csgraph
    csgraph = tg.utils.to_scipy_sparse_matrix(data.edge_index).tocsr()
    
    if cached:
        absolute_path = os.path.dirname(__file__)
        relative_path = f"../data/{data.dataset_name}/rst"
        data_path = os.path.join(absolute_path, relative_path)
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        build_fn = BuildRstCached(path=data_path)
    else:
        build_fn = build_rst
    
    rst_repo.reset()

    import concurrent.futures
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(build_fn, i) for i in list(range(nr_rsts))] # type: ignore
        for future in tqdm.tqdm(futures):
            rst, rpg = future.result()
            rst_repo.add_rst(rst)
            rst_repo.add_rpg(rpg)
    
    return rst_repo