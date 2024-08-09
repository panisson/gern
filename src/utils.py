"""
Utils package with classes and functions that are independent 
of computational backend (pytorch or tensorflow).
"""
import random
from collections import defaultdict
import networkx as nx
import numpy as np
import numba as nb
from numba import typed

@nb.njit
def set_seed(value):
    np.random.seed(value)
    random.seed(value)

@nb.experimental.jitclass(
    [('item_to_position', nb.types.DictType(nb.int64, nb.int64)),
     ('items', nb.types.ListType(nb.types.int64))])
class ListDict():
    """
    Class implementation of a hybrid list dictionary.
    Offers the possibility of going from item value to its position in O(1),
    and from position to value in O(1).
    """
    def __init__(self, iterable=None):
        self.item_to_position = typed.Dict.empty(nb.int64, nb.int64)
        self.items = typed.List.empty_list(nb.int64)
        if iterable is not None:
            for i in iterable:
                self.add_item(i)

    def add_item(self, item):
        """
        Add an item to this data structure.
        """
        if item in self.item_to_position:
            return
        self.items.append(item)
        self.item_to_position[item] = len(self.items)-1

    def remove_item(self, item):
        """
        Remove an item from this data structure.
        """
        position = self.item_to_position[item]
        del self.item_to_position[item]
        last_item = self.items.pop()
        if position != len(self.items):
            self.items[position] = last_item
            self.item_to_position[last_item] = position

    def index(self, item):
        """
        Return the position of an item in the data structure.
        """
        return self.item_to_position[item]

    def choose_random_item(self):
        """
        Choose a random item from the data structure
        through random choice.
        """
        idx = int(random.random() * len(self.items))
        return self.items[idx]

    def __contains__(self, item):
        return item in self.item_to_position

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, val):
        return self.items[val]


class Graph():
    """
    Utility class for lightweight representation of a graph.
    """
    def __init__(self):
        self.graph = defaultdict(set)

    def add_edge(self, source, target):
        """
        Add an undirected edge with source and target to the graph.
        """
        self.graph[source].add(target)
        self.graph[target].add(source)

    def add_edges(self, edges):
        """
        Add a list of edges to the graph.
        """
        for source, target in edges:
            self.graph[source].add(target)
            self.graph[target].add(source)

    def neighbors(self, node):
        """
        Get the neighbors of node.
        """
        return self.graph[node]

    def nodes(self):
        """
        Get the list of nodes.
        """
        return sorted(self.graph.keys())

    def number_of_nodes(self):
        """
        Returns the number of nodes.
        """
        return len(self.graph)

    def number_of_edges(self):
        """
        Returns the number of edges.
        """
        return sum(len(neighbors) for neighbors in self.graph.values())

    def __len__(self):
        return len(self.graph)

    def __iter__(self):
        return iter(self.graph)

    def is_directed(self):
        """
        Return False if the graph is undirected (always the case).
        """
        return False

    def degree(self):
        """
        Return a dictionary with the node as key and its degree as value.
        """
        return {n:len(self.graph[n]) for n in self.graph}

    def is_tree(self):
        """
        Check if this graph is a tree.
        """
        discovered = set()
        nr_edges = 0
        S = [0]
        while len(S) > 0:
            v = S.pop()
            if v not in discovered:
                discovered.add(v)
                for n in list(self.neighbors(v)):
                    S.append(n)
                    nr_edges += 1
        return len(discovered) - 1 == nr_edges/2

    @staticmethod
    def from_networkx(graph):
        """
        Create a lightweight graph representation from a NetworkX graph.
        """
        g = Graph()
        for s,t in graph.edges():
            g.graph[s].add(t)
            g.graph[t].add(s)
        return g

    @staticmethod
    def from_edge_index(edge_index):
        """
        Create a graph representation from a edge index representation.
        """
        g = Graph()
        for s,t in edge_index.T:#.detach().numpy():
            s, t = int(s), int(t)
            g.graph[s].add(t)
            g.graph[t].add(s)
        return g


def random_spanning_tree(graph, seed):
    'Returns a uniform random spanning tree of graph'
    set_seed(seed)
    random.seed(seed)
    unvisited_vertices = ListDict(typed.List(graph.nodes()))
    rst = Graph()
    first_vtx = unvisited_vertices.choose_random_item()
    unvisited_vertices.remove_item(first_vtx)
    while len(unvisited_vertices) > 0:
        start_vtx = unvisited_vertices.choose_random_item()
        current_vtx = start_vtx
        path = [current_vtx]
        while current_vtx in unvisited_vertices:
            next_vtx = random.choice(list(graph.neighbors(current_vtx)))
            if next_vtx in path: 
                i = path.index(next_vtx) 
                path = path[:i]
            path.append(next_vtx)
            current_vtx = next_vtx

        s = path[0]
        for t in path[1:]:
            rst.add_edge(s, t)
            unvisited_vertices.remove_item(s)
            s = t
    return rst


def rw_random_spanning_tree(graph, max_cycles=-1):
    """
    Returns a uniform random spanning tree of graph
    using the random walk method
    """
    unvisited_vertices = ListDict(typed.List(graph.nodes()))
    rst = Graph()
    current_vtx = unvisited_vertices.choose_random_item()
    unvisited_vertices.remove_item(current_vtx)
    cycles = 0
    while len(unvisited_vertices) > 0:
        cycles += 1
        if max_cycles > 0 and cycles >= max_cycles:
            break
        next_vtx = random.choice(list(graph.neighbors(current_vtx)))
        if next_vtx in unvisited_vertices:
            rst.add_edge(current_vtx, next_vtx)
            unvisited_vertices.remove_item(next_vtx)
        current_vtx = next_vtx
    return rst

def random_spanning_tree_v2(graph, seed, rw_cycles=None):
    'Returns a uniform random spanning tree of graph'
    set_seed(seed)
    random.seed(seed)
    if rw_cycles is None:
        rw_cycles = len(graph)*2
    unvisited_vertices = ListDict(typed.List(graph.nodes()))
    rst = rw_random_spanning_tree(graph, max_cycles=rw_cycles)
    for n in rst.nodes():
        unvisited_vertices.remove_item(n)
    # first_vtx = unvisited_vertices.choose_random_item()
    # unvisited_vertices.remove_item(first_vtx)

    while len(unvisited_vertices) > 0:
        start_vtx = unvisited_vertices.choose_random_item()
        current_vtx = start_vtx
        path = [current_vtx]

        while current_vtx in unvisited_vertices:
            next_vtx = random.choice(list(graph.neighbors(current_vtx)))
            if next_vtx in path: 
                i = path.index(next_vtx) 
                path = path[:i]
            path.append(next_vtx)
            current_vtx = next_vtx

        s = path[0]
        for t in path[1:]:
            rst.add_edge(s, t)
            unvisited_vertices.remove_item(s)
            s = t
    
    assert len(rst) == len(graph.nodes())
    return rst

    # Possible optimization:
    '''
    while len(unvisited_vertices) > 0:
        start_vtx = unvisited_vertices.choose_random_item()
        current_vtx = start_vtx
        path = ListDict([current_vtx])

        while current_vtx in unvisited_vertices:
            next_vtx = random.choice(list(graph.neighbors(current_vtx)))
            if next_vtx in path:
                i = path.index(next_vtx)
                path = ListDict(path.items[:i])
            path.add_item(next_vtx)
            current_vtx = next_vtx

        s = path[0]
        for t in path.items[1:]:
            rst.add_edge(s, t)
            unvisited_vertices.remove_item(s)
            s = t
    return rst
    '''

def nx_random_spanning_tree(graph: nx.Graph): 
    'Returns a uniform random spanning tree of graph'
    unvisited_vertices = ListDict(graph.nodes())
    rst = nx.create_empty_copy(graph) # type: ignore
    first_vtx = unvisited_vertices.choose_random_item()
    unvisited_vertices.remove_item(first_vtx)
    while len(unvisited_vertices) > 0: 
        start_vtx = unvisited_vertices.choose_random_item()
        current_vtx = start_vtx
        path = [current_vtx]
        while current_vtx in unvisited_vertices: 
            next_vtx = random.sample(list(graph.neighbors(current_vtx)), 1)[0]
            if next_vtx in path: 
                i = path.index(next_vtx) 
                path = path[:i]
            path.append(next_vtx)
            current_vtx = next_vtx

        source = path[0]
        for target in path[1:]:
            rst.add_edge(source, target)
            unvisited_vertices.remove_item(source)
            source = target
    return rst

@nb.njit
def sp_depth_first_search(N, indptr, indices, root=None):
    """
    Implement graph linearizationg via depth first search.
    """
    discovered = np.zeros(N, dtype=np.uint8)
    # line_graph = np.empty(graph.number_of_nodes(), dtype=int)

    if root is None:
        root = random.randint(0, N-1)
    line_graph = []
    S = [root]
    while len(S) > 0:
        v = int(S.pop())
        if discovered[v] == 0:
            discovered[v] = 1
            line_graph.append(v)

            neighs = indices[indptr[v]: indptr[v+1]].copy()

            # neighs = list(graph.neighbors(v))
            random.shuffle(neighs)
            for n in neighs:
                # if discovered[n] == 0:
                S.append(n)

    return np.array(line_graph)

def depth_first_search(graph, root=None):
    """
    Implement graph linearizationg via depth first search.
    """
    discovered = np.zeros(graph.number_of_nodes(), dtype=bool)
    # line_graph = np.empty(graph.number_of_nodes(), dtype=int)
    if root is None:
        root = random.choice(list(graph.nodes()))
    line_graph = []
    S = [root]
    while len(S) > 0:
        v = S.pop()
        if discovered[v] == 0:
            discovered[v] = 1
            line_graph.append(v)
            neighs = list(graph.neighbors(v))
            random.shuffle(neighs)
            for n in neighs:
                if discovered[n] == 0:
                    S.append(n)
    return np.array(line_graph)

    discovered = ListDict()
    S = [root]
    while len(S) > 0:
        v = S.pop()
        if v not in discovered:
            discovered.add_item(v)
            neighs = list(graph.neighbors(v))
            random.shuffle(neighs)
            for n in neighs:
                S.append(n)
    return np.array(discovered.items)


def depth_first_search_with_bt(graph, root=None, sort=False):
    """
    Implement graph linearizationg via depth first search
     (with backtracking).
    """
    if root is None:
        root = random.sample(list(graph.nodes()), 1)[0]
    discovered = set()
    discovered_edges = set()
    rpg = []
    S = [root]
    while len(S) > 0:
        v = S.pop()
        rpg.append(v)
        if v not in discovered:
            discovered.add(v)
            undiscovered_neighs = []
            neighs = list(graph.neighbors(v))
            if sort:
                neighs.sort(reverse=True)
            else:
                random.shuffle(neighs)
            for n in neighs:
                if n not in discovered:
                    undiscovered_neighs.append(n)
                elif (v,n) not in discovered_edges:
                    S.append(n)
                discovered_edges.add((v,n))
            S.extend(undiscovered_neighs)

    return rpg


def predict_node(rpg, Q, labels, idx):
    n = len(rpg)
    for j in range(1, len(rpg)):
        if idx+j < n and rpg[idx+j] in Q:
            return labels[rpg[idx+j]]
        if idx-j > 0 and rpg[idx-j] in Q:
            return labels[rpg[idx-j]]
        if idx+j > n and idx-j < 0:
            return None

def predict(rpg, Q, labels):
    pred = []
    rpg = ListDict(rpg)
    Qs = set(list(Q))
    for i, label in enumerate(labels):
        if i in Qs:
            pred.append(label)
        else:
            idx = rpg.item_to_position[i]
            pred_label = predict_node(rpg, Qs, labels, idx)
            pred.append(pred_label)
    return pred

def predict_fast(rpg, Q, labels, max_dist=None):
    Qs = set(list(Q))
    prediction = dict()
    pred_prev = None
    idx_prev = 0
    for i, node in enumerate(rpg):
        if node in Qs:
            prediction[node] = (labels[node],0)
            
            if pred_prev is None:
                pred_prev = labels[node]
                idx_prev = i
                for j in range(i):
                    prediction[rpg[j]] = (pred_prev, i-j)
            
            else:
                
                s, m = idx_prev+1, idx_prev + (i-idx_prev-1)//2 + 1 # preference to last
                # s, m = idx_prev+1, idx_prev + (i-idx_prev)//2 + 1
                for j in range(s, m):
                    prediction[rpg[j]] = (pred_prev, j-idx_prev)
                
                #prediction[rpg[m]].append((pred_prev, j+2-idx_prev))
                pred_prev = labels[node]
                for j in range(m, i):
                    prediction[rpg[j]] = (pred_prev, i-j)
                
            idx_prev = i
                    
    if node not in Qs:
        for j in range(idx_prev+1, i+1):
            prediction[rpg[j]] = (pred_prev, j-idx_prev)
    
    if max_dist is None:
        return [prediction[i][0] for i in range(len(labels))]
    else:
        return [prediction[i][0] if prediction[i][1] <= max_dist else None for i in range(len(labels))]

@nb.jit
def numba_propagate(arr):
    out = arr.copy()
    cnt = np.empty(arr.shape[0], dtype=np.int64)
    fill0 = None
    for idx in range(out.shape[0]):
        if not np.isnan(out[idx]):
            fill0 = out[idx]
            break
        cnt[idx] = 100000
    out[:idx] = fill0
    for idx in range(idx, out.shape[0]):
        if np.isnan(arr[idx]):
            out[idx] = out[idx - 1]
            cnt[idx] = cnt[idx - 1] + 1
        else:
            cnt[idx] = 0

    out_ref = arr.copy()
    cnt_ref = np.empty(arr.shape[0], dtype=np.int64)
    fill0 = None
    for idx in range(out_ref.shape[0]-1, 0, -1):
        if not np.isnan(out_ref[idx]):
            fill0 = out_ref[idx]
            break
        cnt_ref[idx:] = 100000
    out_ref[idx:] = fill0
    for idx in range(idx, -1, -1):
        if np.isnan(arr[idx]):
            out_ref[idx] = out_ref[idx + 1]
            cnt_ref[idx] = cnt_ref[idx + 1] + 1
        else:
            cnt_ref[idx] = 0
    
    mask = cnt<cnt_ref
    return np.where(mask, out, out_ref), np.where(mask, cnt, cnt_ref)

@nb.jit
def propagate(rpg, Q, labels, max_dist=None):
    arr = np.empty(labels.shape[0])
    arr[:] = np.nan
    arr[Q] = labels[Q]
    arr = arr[rpg]
    pred, dist = numba_propagate(arr)
    if max_dist is not None:
        pred[dist>max_dist] = np.nan
        
    pred_rpg = np.empty(len(rpg))
    dist_rpg = np.empty(len(rpg))
    for i, n in enumerate(rpg):
        pred_rpg[n] = pred[i]
        dist_rpg[n] = dist[i]
    return pred_rpg, dist_rpg

@nb.jit
def propagate_ohe(rpg, Q, labels_ohe):
    arr = np.empty_like(labels_ohe)
    arr[:,:] = np.nan
    arr[Q] = labels_ohe[Q]
    arr = arr[rpg]
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i,j] == 0: arr[i, j] = np.nan
    pred = np.empty_like(labels_ohe)
    dist = np.empty_like(labels_ohe, dtype=np.int64)
    for i in range(arr.shape[1]):
        pred[:, i], dist[:, i] = numba_propagate(arr[:, i])
    
    dist_rpg = np.empty_like(dist)
    for i, n in enumerate(rpg):
        dist_rpg[n] = dist[i]
    return dist_rpg

def predict_node_steps(rpg, Q, labels, idx):
    n = len(rpg)
    for j in range(1, len(rpg)):
        if idx+j < n and rpg[idx+j] in Q:
            return labels[rpg[idx+j]], j
        elif idx-j > 0 and rpg[idx-j] in Q:
            return labels[rpg[idx-j]], j
        elif idx+j > n and idx-j < 0:
            return None, j

def predict_bt(rpg, Q, labels):
    pred = []
    item_to_position = defaultdict(list)
    for i, node in enumerate(rpg):
        item_to_position[node].append(i)
        
    Qs = set(list(Q))
    for i in range(len(labels)):
        if i in Qs:
            pred.append(labels[i])
        else:
            steps = len(rpg)
            pred_label = None
            #if i==52: print(item_to_position[i])
            for idx in item_to_position[i]:
                pred_label_idx, steps_idx = predict_node_steps(rpg, Qs, labels, idx)
                #if i==52: print(pred_label_idx, steps_idx)
                if steps_idx < steps and pred_label_idx is not None:
                    #if i==52: print(pred_label, pred_label_idx)
                    pred_label = pred_label_idx
                    steps = steps_idx
            pred.append(pred_label)
    return pred


def assign_lower_d(prediction, n, pred, d):
    if n not in prediction:
        prediction[n] = pred, d
    else:
        if prediction[n][1] > d:
            prediction[n] = pred, d

# def predict_bt_fast(rpg, Q, labels):
#     Qs = set(list(Q))
#     prediction = dict()
#     pred_prev = None
#     idx_prev = 0
#     for i, node in enumerate(rpg):
#         if node in Qs:
#             prediction[node] = labels[node],
            
#             if pred_prev is None:
#                 pred_prev = labels[node]
#                 idx_prev = i
#                 for j in range(i):
#                     prediction[rpg[j]] = pred_prev, i-j
            
#             else:
                
#                 # s, m = idx_prev+1, idx_prev + (i-idx_prev-1)//2 + 1 # preference to last
#                 s, m = idx_prev+1, idx_prev + (i-idx_prev)//2 + 1
#                 for j in range(s, m):
#                     assign_lower_d(prediction, rpg[j], pred_prev, j-idx_prev)
                
#                 pred_prev = labels[node]
#                 for j in range(m, i):
#                     assign_lower_d(prediction, rpg[j], pred_prev, i-j)
                
#             idx_prev = i
#             #pred_prev = labels[node]
                    
#     if node not in Qs:
#         for j in range(idx_prev+1, i+1):
#             assign_lower_d(prediction, rpg[j], pred_prev, j-idx_prev)
    
#     return [prediction[i][0] for i in range(len(labels))]


def predict_bt_fast(rpg, Q, labels):
    Qs = set(list(Q))
    prediction = defaultdict(list)
    pred_prev = None
    idx_prev = 0
    for i, node in enumerate(rpg):
        if node in Qs:
            prediction[node] = [(labels[node],0)]
            
            if pred_prev is None:
                pred_prev = labels[node]
                idx_prev = i
                for j in range(i):
                    prediction[rpg[j]].append((pred_prev, i-j))
            
            else:
                
                s, m = idx_prev+1, idx_prev + (i-idx_prev-1)//2 + 1 # preference to last
                # s, m = idx_prev+1, idx_prev + (i-idx_prev)//2 + 1
                for j in range(s, m):
                    prediction[rpg[j]].append((pred_prev, j-idx_prev))
                
                #prediction[rpg[m]].append((pred_prev, j+2-idx_prev))
                pred_prev = labels[node]
                for j in range(m, i):
                    prediction[rpg[j]].append((pred_prev, i-j))
                
            idx_prev = i
                    
    if node not in Qs:
        for j in range(idx_prev+1, i+1):
            prediction[rpg[j]].append((pred_prev, j-idx_prev))
    
    res = []
    for i in range(len(labels)):
        mind = min([t[1] for t in  prediction[i]])
        preds = [t[0] for t in  prediction[i] if t[1]==mind]
        res.append(np.bincount(preds).argmax())
    return res


def test_graph():
    graph = Graph()
    graph.add_edge(0, 1)
    
    assert(graph.neighbors(0) == {1})
    assert(graph.neighbors(1) == {0})
    
    graph.add_edge(0, 2)
    assert(graph.neighbors(0) == {1, 2})
    
    graph.add_edge(2, 1)
    assert(graph.neighbors(2) == {0, 1})

def index_to_mask(index, size):
    mask = np.zeros(size, dtype=bool)
    mask[index] = 1
    return mask

def train_test_split(N, train_size=0.1, val_size=None, seed=0):
    np.random.seed(seed)

    if val_size is None:
        val_size = (1-train_size)/2

    idx_rnd = np.arange(N)
    np.random.shuffle(idx_rnd)

    train_samples = int(N * train_size)
    val_samples = int(N * val_size)

    train_mask = index_to_mask(idx_rnd[:train_samples], N)
    val_mask = index_to_mask(idx_rnd[train_samples:train_samples+val_samples], N)
    test_mask = index_to_mask(idx_rnd[train_samples+val_samples:], N)

    return train_mask, val_mask, test_mask

class TrainTestSplit():

    def __init__(self, train_size, val_size=None) -> None:
        self.train_size = train_size
        self.val_size = val_size

    def __call__(self, y, seed):
        return train_test_split(y.shape[0], seed=seed,
                                train_size=self.train_size, val_size=self.val_size)

def random_planetoid_split(y, num_classes, num_train_per_class=20,
                           num_val=500, num_test=None, seed=0):
    np.random.seed(seed)

    indices = []
    for i in range(num_classes):
        index = np.where(y == i)[0]
        index = index[np.random.permutation(index.shape[0])]
        indices.append(index)

    train_index = np.hstack([i[:num_train_per_class] for i in indices])

    rest_index = np.hstack([i[num_train_per_class:] for i in indices])
    rest_index = rest_index[np.random.permutation(rest_index.shape[0])]
    
    train_mask = index_to_mask(train_index, size=y.shape[0])
    num_val = int(rest_index.shape[0]*0.1)
    val_mask = index_to_mask(rest_index[:num_val], size=y.shape[0])
    test_mask = index_to_mask(rest_index[num_val:], size=y.shape[0])
    
    return train_mask, val_mask, test_mask

class PlanetoidSplit():

    def __init__(self, num_classes, num_train_per_class):
        self.num_classes = num_classes
        self.num_train_per_class = num_train_per_class
    
    def __call__(self, y, seed):
        return random_planetoid_split(y, self.num_classes, self.num_train_per_class,
                                      num_val=500, seed=seed)

def k_hop_subgraph(Q, num_hops, graph):
    subgraph = Graph()
    queue = list(Q)
    visited = set()
    for hop in range(num_hops):
        next_hop_queue = []
        for n in queue:
            for neighbour in graph.neighbors(n):
                subgraph.add_edge(n, neighbour)
                if neighbour not in visited:
                    visited.add(neighbour)
                    next_hop_queue.append(neighbour)
                    
        queue = next_hop_queue
    
    return subgraph
    
def relabel_nodes(graph):
    relabel = {}
    new_label = 0
    for node in graph.nodes():
        relabel[node] = new_label
        new_label += 1
        
    relabeled_graph = Graph()
    relabeled_graph.add_edges([(relabel[i], relabel[j]) for i, j in graph.edges])
    return relabeled_graph, relabel

def write_report(experimental_data, results_path, dataset, experiment, **kwargs):
    """
    Write the results in experimental_data to a csv file.
    """
    
    #BUG: pandas must be loaded after obg...
    import pandas as pd
    import hashlib

    df_list = []
    for data in experimental_data:

        df = pd.DataFrame(data['results'])
        df['dataset'] = dataset
        df['experiment'] = experiment
        for k, v in kwargs.items():
            df[k] = str(v)
        for k, v in data.items():
            if k != 'results':
                df[k] = str(v)

        df_list.append(df)
    df = pd.concat(df_list)

    h = hashlib.sha1(pd.util.hash_pandas_object(df).values).hexdigest()  # type: ignore
    df.to_csv(f'{results_path}/results_{dataset}_{experiment}_{h}.csv')

    def select_max_val(subdf):
        idx = subdf['accuracy_val'].argmax()
        return subdf.iloc[idx]

    df = df.groupby(['num_train_per_class', 'method', 'seed']).apply(select_max_val).reset_index(drop=True)
    df = df.groupby(['num_train_per_class', 'method']).accuracy_test.apply(
        lambda subdf: f"{(subdf.mean()*100):.2f}±{(subdf.std()/np.sqrt(subdf.shape[0])*100):.2f}")\
        .to_frame()

    print(df)