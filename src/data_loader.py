import os
import pickle as pkl
import random
import numpy as np
import torch
import torch_geometric as tg
from torch_geometric import utils as tgu
import torch_geometric.transforms as T
from torch_geometric.data import Data
import networkx as nx

def load_cora():

    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToUndirected()
    ])
    dataset = tg.datasets.Planetoid(root='./dataset', name='cora', transform=transform)
    data = dataset[0]

    # data = T.LargestConnectedComponents()(data)

    G = tgu.to_networkx(data, to_undirected=True,
                        node_attrs=['x', 'y', 'train_mask', 'val_mask', 'test_mask'])
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    G = nx.convert_node_labels_to_integers(G) # type: ignore
    data = tgu.from_networkx(G)

    data.num_classes = dataset.num_classes
    data.dataset_name = "cora"
    return data

def load_pubmed():

    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToUndirected()
    ])
    dataset = tg.datasets.Planetoid(root='./dataset', name='pubmed', transform=transform)
    data = dataset[0]

    absolute_path = os.path.dirname(__file__)
    edge_resistance_path = f'{absolute_path}/dataset/pubmed/pubmed_edge_resistance.npz'
    edge_resistance = np.load(edge_resistance_path)['edge_resistance']
    data.edge_resistance = torch.tensor(edge_resistance, dtype=torch.float32)

#     G = tg.utils.to_networkx(data, to_undirected=True, node_attrs=['x', 'y'])
#     largest_cc = max(nx.connected_components(G), key=len)
#     G = G.subgraph(largest_cc).copy()
#     G = nx.convert_node_labels_to_integers(G)

#     data = tg.utils.from_networkx(G)
    data.num_classes = dataset.num_classes
    data.dataset_name = "pubmed"
    return data

def load_ogbn_arxiv():
    from ogb.nodeproppred import PygNodePropPredDataset

    absolute_path = os.path.dirname(__file__)
    root_path = f"{absolute_path}/dataset"
    dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                    transform=T.ToUndirected(), root=root_path)

    data = dataset[0]
    edge_index = data.edge_index
    data = T.ToSparseTensor()(data)
    data.edge_index = edge_index
    data.num_classes = dataset.num_classes
    data.y = data.y.squeeze(1)
    data.dataset_name = "ogbn-arxiv"

    split_idx = dataset.get_idx_split()
    train_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
    val_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
    test_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
    train_mask[split_idx['train']] = True
    val_mask[split_idx['valid']] = True
    test_mask[split_idx['test']] = True
    data.train_mask, data.val_mask, data.test_mask = (
        train_mask, val_mask, test_mask
    )

    # load edge resistance
    edge_resistance_path = f'{root_path}/ogbn_arxiv/ogbn_arxiv_edge_resistance.npz'
    if os.path.exists(edge_resistance_path):
        edge_resistance = np.load(edge_resistance_path)['edge_resistance']
        data.edge_resistance = torch.tensor(edge_resistance, dtype=torch.float32)

    return data

def load_ogbn_mag():
    from ogb.nodeproppred import PygNodePropPredDataset

    absolute_path = os.path.dirname(__file__)
    root_path = f"{absolute_path}/dataset"
    dataset = PygNodePropPredDataset(name='ogbn-mag',
                        transform=T.ToUndirected(), root=root_path)

    rel_data = dataset[0]
        # We are only interested in paper <-> paper relations.
    data = Data(
        x=rel_data.x_dict['paper'],
        edge_index=rel_data.edge_index_dict[('paper', 'cites', 'paper')],
        y=rel_data.y_dict['paper'])
    data = T.ToUndirected()(data)

    split_idx = dataset.get_idx_split()
    train_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
    val_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
    test_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
    train_mask[split_idx['train']['paper']] = True
    val_mask[split_idx['valid']['paper']] = True
    test_mask[split_idx['test']['paper']] = True
    data.train_mask, data.val_mask, data.test_mask = (
        train_mask, val_mask, test_mask
    )

    data = T.LargestConnectedComponents()(data)
    data.num_classes = dataset.num_classes
    data.y = data.y.squeeze(1)
    data.dataset_name = "ogbn-mag"

    return data

def load_flickr():
    absolute_path = os.path.dirname(__file__)
    root_path = f"{absolute_path}/dataset/flickr"

    dataset = tg.datasets.Flickr(root_path)
    data = dataset[0]
    data = T.ToUndirected()(data)
    data = T.LargestConnectedComponents()(data) # type: ignore
    data.num_classes = dataset.num_classes
    data.dataset_name = "flickr"
    return data

def load_yelp():
    absolute_path = os.path.dirname(__file__)
    root_path = f"{absolute_path}/dataset/yelp"

    dataset = tg.datasets.Yelp(root_path)
    data = dataset[0]
    data = T.ToUndirected()(data)
    data = T.LargestConnectedComponents()(data) # type: ignore
    data.num_classes = dataset.num_classes
    data.dataset_name = "yelp"
    return data

def load_reddit():
    absolute_path = os.path.dirname(__file__)
    root_path = f"{absolute_path}/dataset/reddit"

    dataset = tg.datasets.Reddit(root_path)
    data = dataset[0]
    data = T.ToUndirected()(data)
    data = T.LargestConnectedComponents()(data) # type: ignore
    
    # create sparse tensor instance
    edge_index = data.edge_index
    data = T.ToSparseTensor()(data)
    data.edge_index = edge_index
    
    data.num_classes = dataset.num_classes
    data.dataset_name = "reddit"
    return data

def load_reddit2():
    absolute_path = os.path.dirname(__file__)
    root_path = f"{absolute_path}/dataset/reddit2"

    dataset = tg.datasets.Reddit2(root_path)
    data = dataset[0]
    data = T.ToUndirected()(data)
    data = T.LargestConnectedComponents()(data) # type: ignore
    data.num_classes = dataset.num_classes
    data.dataset_name = "reddit2"
    return data

def load_aminer():
    absolute_path = os.path.dirname(__file__)
    root_path = f"{absolute_path}/dataset/aminer"
    
    dataset_str = 'aminer'
    adj = pkl.load(open(os.path.join(root_path, "{}.adj.sp.pkl".format(dataset_str)), "rb"))
    features = pkl.load(
        open(os.path.join(root_path, "{}.features.pkl".format(dataset_str)), "rb"))
    labels = pkl.load(
        open(os.path.join(root_path, "{}.labels.pkl".format(dataset_str)), "rb"))
    # random_state = np.random.RandomState(split_seed)
    # idx_train, idx_val, idx_test = get_train_val_test_split(
    #     random_state, labels, train_examples_per_class=20, val_examples_per_class=30)
    # idx_unlabel = np.concatenate((idx_val, idx_test))
    # features = col_normalize(features)

    edge_index = tgu.from_scipy_sparse_matrix(adj)[0]
    data = Data(x=torch.tensor(features, dtype=torch.float32), 
                edge_index=edge_index, y=torch.tensor(labels.argmax(axis=-1)))
    
    # include sparse tensor for faster computation
    data.adj_t = tg.typing.SparseTensor(row=edge_index[0], col=edge_index[1])

    # load edge resistance
    edge_resistance_path = f'{root_path}/aminer_edge_resistance.npz'
    if os.path.exists(edge_resistance_path):
        edge_resistance = np.load(edge_resistance_path)['edge_resistance']
        data.edge_resistance = torch.tensor(edge_resistance, dtype=torch.float32)

    data.num_classes = labels.shape[1]
    data.dataset_name = "aminer"
    return data

def load_data(dataset_str='cora'):
    if dataset_str=='cora':
        return load_cora()
    if dataset_str == 'aminer':
        return load_aminer()

def train_test_split(N, rate=0.1, seed=0):
    np.random.seed(seed)
    random.seed(seed)

    idx_rnd = np.arange(N)
    np.random.shuffle(idx_rnd)
    
    q = int(N * rate)
    Q = idx_rnd[:q]
    
    train_mask = np.zeros(N, dtype=bool)
    train_mask[Q] = 1
    test_mask = np.ones(N, dtype=bool)
    test_mask[Q] = 0
    
    return Q, train_mask, test_mask

def load_ogbn_products():
    from ogb.nodeproppred import PygNodePropPredDataset

    absolute_path = os.path.dirname(__file__)
    root_path = f"{absolute_path}/dataset"
    dataset = PygNodePropPredDataset(name='ogbn-products',
                    transform=T.ToUndirected(), root=root_path)

    data = dataset[0]

    split_idx = dataset.get_idx_split()
    train_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
    val_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
    test_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
    train_mask[split_idx['train']] = True
    val_mask[split_idx['valid']] = True
    test_mask[split_idx['test']] = True
    data.train_mask, data.val_mask, data.test_mask = (
        train_mask, val_mask, test_mask
    )

    post_transform = T.Compose([
        T.RemoveIsolatedNodes(),
        T.LargestConnectedComponents(),
    ]) 
    data = post_transform(data)

    edge_index, _ = tgu.remove_self_loops(data.edge_index)
    data.edge_index = edge_index
    data = T.ToSparseTensor()(data)
    data.edge_index = edge_index
    data.num_classes = dataset.num_classes
    data.y = data.y.squeeze(1)
    data.dataset_name = "ogbn-products"

    # data.adj_t = tg.typing.SparseTensor(row=data.edge_index[0], col=data.edge_index[1])

    # load edge resistance
    edge_resistance_path = f'{root_path}/ogbn_products/ogbn_products_edge_resistance.npz'
    if os.path.exists(edge_resistance_path):
        edge_resistance = np.load(edge_resistance_path)['edge_resistance']
        data.edge_resistance = torch.tensor(edge_resistance, dtype=torch.float32)

    return data