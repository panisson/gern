import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric as tg
from torch_geometric.loader import NeighborLoader
import torch_geometric.utils as tgu
from torch_geometric.utils import k_hop_subgraph
from sklearn import metrics

import tqdm
import utils
from models import SAGE, SAGE_OGB, GCN
import rst_utils as rstu

rst_repo = rstu.rst_repo

global device
def set_device(device_nr):
    global device
    device = torch.device(f'cuda:{device_nr}' if torch.cuda.is_available() else 'cpu')
    rst_repo.to(device)

def verify_train_size(data, num_train_per_class):
    pass
    # counts = np.bincount(data.y.detach().cpu().numpy())
    # for c in counts:
    #     if num_train_per_class > c*0.8:
    #         print(counts)
    #         raise Exception(f"One class has only {c} samples.")

def to_tensor(x):
    if torch.is_tensor(x):
        return x
    else:
        return torch.tensor(x)
    
def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    else:
        return x

@torch.no_grad()
def predict(model, data):
    model.eval()
    edges = data.adj_t if hasattr(data, 'adj_t') and  data.adj_t is not None else data.edge_index
    return model(data.x, edges)

def accuracy(pred, y):
    acc = pred.eq(y).sum().item() / y.shape[0]
    return acc

@torch.no_grad()
def accs_and_losses(model, y, log_probs, *masks):
    model.eval()
    scores = []
    for mask in masks:
        loss = F.nll_loss(log_probs[mask], y[mask])
        pred = log_probs[mask].max(1)[1]
        acc = accuracy(pred, y[mask])
        scores.append({'loss':loss.item(), 'accuracy': acc})
    return scores

@torch.no_grad()
def test_full(model, data, *masks):
    log_probs = predict(model, data)
    return accs_and_losses(model, data.y, log_probs, *masks)

@torch.no_grad()
def test_rpg(model, data, *masks):
    model.eval()
    
    nr_rpgs = len(rst_repo.rpgs)
    edge_index = rst_repo.get_edge_index(0)
    log_probs = model(data.x, edge_index)
    for rpg_index in range(1, nr_rpgs):
        edge_index = rst_repo.get_edge_index(rpg_index)
        log_probs += model(data.x, edge_index)
    log_probs /= nr_rpgs
        
    return accs_and_losses(model, data, log_probs, *masks)

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']

def reduce_lr(optimizer, factor):
    for i, param_group in enumerate(optimizer.param_groups):
        old_lr = float(param_group['lr'])
        new_lr = old_lr * factor
        param_group['lr'] = new_lr

class ExperimentRunner():

    def __init__(self, experiment, data, split, verbose=0, **vargs):
        self.experiment = experiment
        self.data = data
        self.split = split
        self.verbose = verbose
        self.vargs = vargs
    
    def run_many_experiments(self, data, exp_name, nr_experiments, num_train_per_class):
        vargs = self.vargs
        hidden_channels = vargs["hidden_channels"]
        num_layers = vargs["num_layers"]
        dropout = vargs["dropout"]
        lr = vargs["lr"]
        weight_decay = vargs["wd"]
        epochs = vargs["epochs"]
        min_epochs = vargs["min_epochs"]
        patience = vargs["patience"]
        val_step = vargs["val_step"]
        start_seed = vargs["start_seed"]
        use_bn = vargs["use_bn"]

        def create_model():
            if vargs["use_gcn"]:
                model = GCN(data.num_features, hidden_channels, data.num_classes,
                        num_layers=num_layers, dropout=dropout, use_bn=use_bn).to(device)
            elif vargs["use_sage"]:
                model = SAGE_OGB(data.num_features, hidden_channels, data.num_classes,
                        num_layers=num_layers, dropout=dropout, use_bn=use_bn).to(device)
            else:
                model = SAGE(data.num_features, hidden_channels, data.num_classes,
                        num_layers=num_layers, dropout=dropout, use_bn=use_bn).to(device)
            return model

        for seed in tqdm.tqdm(range(start_seed, nr_experiments+start_seed),
                              position=0, leave=True):
            tg.seed_everything(seed)
            model = create_model() #model_cls(data.num_features, hidden_channels, data.num_classes,
                        # num_layers=num_layers, dropout=dropout).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            experiment = self.experiment(data=data, split=self.split, 
                                         verbose=self.verbose, **vargs)
            results = experiment.run(model, optimizer, seed=seed,
                                epochs=epochs, min_epochs=min_epochs,
                                patience=patience, val_step=val_step)
            
            yield {
                # 'method': exp_name,
                # 'num_train_per_class': num_train_per_class,
                'seed': seed,
                'results': results
            }

# ------------------ Experiments ----------------

class Experiment():
    
    def __init__(self, data, split, verbose=0, **vargs):
        self.data = data
        self.split = split
        self.verbose = verbose
        self.vargs = vargs
        
    def setup(self, model, optimizer, seed):
        self.model = model.to(device)
        self.optimizer = optimizer

        if self.split is None:
            # public split
            self.train_mask, self.val_mask, self.test_mask = \
                self.data.train_mask, self.data.val_mask, self.data.test_mask
        else:
            y = to_numpy(self.data.y)
            # if len(y.shape) > 1:
            #     y = self.data.y.argmax(axis=-1)
            self.train_mask, self.val_mask, self.test_mask = self.split(y, seed)

        self.train_mask, self.val_mask, self.test_mask = (
            to_tensor(self.train_mask),#.to(device),
            to_tensor(self.val_mask),#.to(device),
            to_tensor(self.test_mask)#.to(device)
        )

        self.train_idx = torch.where(self.train_mask)[0]
        self.val_idx = torch.where(self.val_mask)[0]
        self.test_idx = torch.where(self.test_mask)[0]

        if self.verbose:
            print(self.split)
            print(f"Split: {self.train_mask.sum()} train, "
                  f"{self.val_mask.sum()} val, {self.test_mask.sum()} test.")
        
    def train_step_args(self):
        return {}
    
    def train_lr(self, data, model, optimizer, learning_rate,
            #  train_loader, subgraph_loader, device,
             epochs=1000, min_epochs=100, patience=10, val_step=1,
             **train_step_args):
    
        results = []
        for g in optimizer.param_groups:
            g['lr'] = learning_rate

        min_loss = np.inf
        count = 0
        for epoch in range(1, epochs+1):
            before_train = time.time()
            self.train_step(epoch, model, optimizer, data, 
                            self.train_idx, **train_step_args)
            train_time = time.time() - before_train

            if epoch > min_epochs and epoch % val_step == 0:

                before_val = time.time()
                scores = self.test_step(model, data, 
                    self.train_idx, self.val_idx, self.test_idx)
                val_loss = scores[1]["loss"]
                val_time = time.time() - before_val

                report = {'epoch': epoch,
                          'train_time': train_time,
                          'val_time': val_time}
                for i, mask_name in enumerate(['train', 'val', 'test']):
                    for k, v in scores[i].items():
                        report[f'{k}_{mask_name}'] = v
                results.append(report)

                if val_loss < min_loss:
                    min_loss = val_loss
                    count = 0
                count += 1
                if count > patience:
                    break

                if self.verbose and epoch % self.verbose == 0:
                    # scores = \
                    #     self.test_step(self.model, self.data, 
                    #                    self.train_idx, self.val_mask, self.test_mask)
                    train_acc = scores[0]["accuracy"]
                    val_acc = scores[1]["accuracy"]
                    test_acc = scores[2]["accuracy"]
                    print(f'LR: {learning_rate}, Epoch: {epoch:03d}, '
                          f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
        return results
    
    def train(self, epochs=1000, min_epochs=100, patience=10, val_step=1,
              **kwargs):

        if self.verbose:
            print("Training...")

        self.data = self.data.to(device)
        results = []

        train_args = self.train_step_args()
        kwargs.update(train_args)

        # for learning_rate in [1e-2, 1e-3, 1e-4, 1e-5]:
        for learning_rate in np.logspace(-2, -4, 5):

            res_lr = self.train_lr(self.data, self.model, self.optimizer, learning_rate,
                # self.train_loader, self.subgraph_loader, device,
                epochs=epochs, min_epochs=min_epochs, patience=patience, val_step=val_step,
                **kwargs)
            
            results.extend(res_lr)
    
        return results
    
    # def eval(self, test_step):
    #     train_acc, test_acc, _, _ = test_step(self.model, self.data, self.train_mask, self.test_mask)
    #     print(f'Train: {train_acc:.4f}, Test: {test_acc:.4f}')
        
    def run(self, model, optimizer, seed,
            epochs=1000, min_epochs=100, patience=10, val_step=1, **kwargs):
        self.setup(model, optimizer, seed)
        results = self.train(epochs=epochs, min_epochs=min_epochs, 
                             patience=patience, val_step=val_step, **kwargs)
        # scores = self.test_step(self.model, self.data, self.test_mask)[0]
        return results

    def train_step(self, epoch, model, optimizer, data, train_idx, **train_step_args):
        raise NotImplementedError()
    
    def test_step(self, model, data, *masks):
        raise NotImplementedError()

class PygExperiment(Experiment):

    def __init__(self, data, split, verbose=0, **vargs):
        super().__init__(data, split, verbose, **vargs)

        self.batch_inference = self.vargs.get('batch_inference', False)

    def setup(self, model, optimizer, seed):
        super().setup(model, optimizer, seed)

        if self.batch_inference:

            self.data = self.data.to('cpu')

            self.data = self.data.to(device, 'x', 'y')
            # self.data = self.data.to(device)
            # self.train_idx = to_tensor(self.train_idx)

            kwargs = {'batch_size': 1024, 'num_workers': 4, 'persistent_workers': True}
            # kwargs = {'batch_size': 1024, 'num_workers': 0}
            # train_loader = NeighborLoader(self.data, input_nodes=self.train_idx,
            #                             num_neighbors=[25, 10], shuffle=True, **kwargs)

            self.loader = NeighborLoader(copy.copy(self.data), input_nodes=None,
                                    num_neighbors=[-1], shuffle=False, **kwargs)
            # No need to maintain these features during evaluation:
            del self.loader.data.x, self.loader.data.y
            # Add global node index information.
            self.loader.data.num_nodes = self.data.num_nodes
            self.loader.data.n_id = torch.arange(self.data.num_nodes)
        else:
            self.data = self.data.to(device)
    
    def train_step_args(self):
        return {}

    def test_step(self, model, data, *masks):
        model.eval()
        if self.batch_inference:
            log_probs = model.inference(data.x, self.loader, verbose=self.verbose)
            return accs_and_losses(model, data.y.cpu(), log_probs, *masks)
        else:
            log_probs = predict(model, data)
        
        return accs_and_losses(model, data.y, log_probs, *masks)

class FULL(PygExperiment):

    def train_step(self, epoch, model, optimizer, data, train_idx):
        model.train()
        optimizer.zero_grad()
        edges = data.adj_t if hasattr(data, 'adj_t') and data.adj_t is not None else data.edge_index
        F.nll_loss(model(data.x, edges)[train_idx], data.y[train_idx]).backward()
        optimizer.step()

class RST(PygExperiment):

    def train_step(self, epoch, model, optimizer, data, train_idx):
        rst = rst_repo.rsts[epoch%len(rst_repo.rsts)]
        edge_index_rst = torch.concat([torch.tensor(list(rst.edges)), 
                                    torch.tensor([(t,s) for (s,t) in rst.edges])]).T.to(device)
        
        model.train()
        optimizer.zero_grad()
        F.nll_loss(model(data.x, edge_index_rst)[train_idx], data.y[train_idx]).backward()
        optimizer.step()

class RPG(PygExperiment):

    def train_step(self, epoch, model, optimizer, data, train_idx):
        rpg_index = epoch%len(rst_repo.rpgs)
        edge_index_rpg = rst_repo.get_edge_index(rpg_index)

        # train_idx = self.train_idx.to(device)

        model.train()
        optimizer.zero_grad()

        num_hops = len(model.convs) #+ 1
        if num_hops > 10: # select subgraph only if hops <= 10
            
            loss = F.nll_loss(model(data.x, edge_index_rpg)[train_idx], data.y[train_idx])
        else:
            Q = train_idx #torch.where(train_idx)[0]
            
            subset, edge_index_subgraph, index_map, edge_mask = k_hop_subgraph(
                Q, num_hops=num_hops, 
                edge_index=edge_index_rpg, relabel_nodes=True)
            x = data.x[subset]
            loss = F.nll_loss(model(x, edge_index_subgraph)[index_map], data.y[Q])
        
        loss.backward()
        optimizer.step()

class RANDOM(PygExperiment):

    def setup(self, model, optimizer, seed):
        super().setup(model, optimizer, seed)

        num_hops = len(model.convs)
        self.train_loader = NeighborLoader(self.data,
                input_nodes=to_tensor(self.train_idx),
                num_neighbors=[2]*num_hops, shuffle=True, batch_size=1024)

        # self.data = self.data.to('cpu')
        # self.data = self.data.to(device, 'x', 'y')

        # kwargs = {'batch_size': 1024, 'num_workers': 4, 'persistent_workers': True}
        # # kwargs = {'batch_size': 1024, 'num_workers': 0}
        # self.train_loader = NeighborLoader(copy.copy(self.data),
        #         input_nodes=to_tensor(self.train_idx),
        #         num_neighbors=[25, 10], shuffle=True, **kwargs)
    
    def train_step(self, epoch, model, optimizer, data, train_idx):

        model.train()

        for batch in self.train_loader:
            
            optimizer.zero_grad()
            y = batch.y[:batch.batch_size]
            y_hat = model(batch.x, batch.edge_index.to(device))[:batch.batch_size]
            loss = F.nll_loss(y_hat, y)
            loss.backward()
            optimizer.step()

def build_edge_resistance(data):
    laplacian = tgu.get_laplacian(data.edge_index)
    sp_laplacian = tg.typing.SparseTensor(row=laplacian[0][0],
                                            col=laplacian[0][1],
                                            value=laplacian[1])
    L_pinv = torch.linalg.pinv(sp_laplacian.to_dense())
    return [(L_pinv[i][i] + L_pinv[j][j] - L_pinv[i][j] - L_pinv[j][i]).item()
                        for i, j in data.edge_index.T]

class GERN(PygExperiment):

    def setup(self, model, optimizer, seed):
        super().setup(model, optimizer, seed)

        if hasattr(self.data, 'edge_resistance'):
            assert self.data.edge_resistance.size(0) == self.data.edge_index.size(1)
            self.edge_resistance = self.data.edge_resistance.to(self.data.edge_index.device)

        else:
            if self.verbose:
                print('Calculating pseudoinverse of laplacian...')
            edge_resistance = build_edge_resistance(self.data)
            self.edge_resistance = torch.tensor(edge_resistance).to(self.data.edge_index.device)

        # self.data = self.data.to('cpu')
        # self.data = self.data.to(device, 'x', 'y')

        # kwargs = {'batch_size': 1024, 'num_workers': 4, 'persistent_workers': True}
        # # kwargs = {'batch_size': 1024, 'num_workers': 0}
        # self.train_loader = NeighborLoader(copy.copy(self.data),
        #         input_nodes=to_tensor(self.train_idx),
        #         num_neighbors=[25, 10], shuffle=True, **kwargs)
    
    def train_step(self, epoch, model, optimizer, data, train_idx):
        train_idx = self.train_idx.to(device)

        model.train()
        optimizer.zero_grad()
        # edges = data.adj_t if hasattr(data, 'adj_t') and data.adj_t is not None else data.edge_index
        edges = data.edge_index
        out = model(data.x, edges, self.edge_resistance)
        loss = F.nll_loss(out[train_idx], data.y[train_idx])
        loss.backward()
        optimizer.step()

class RPG_LP2(PygExperiment):
    def train_step(self, epoch, model, optimizer, data, train_idx):
        rst = rst_repo.rsts[epoch%len(rst_repo.rsts)]
        edge_index_rst = torch.concat(
            [torch.tensor(list(rst.edges)), 
            torch.tensor([(t,s) for (s,t) in rst.edges])]).T.to(device)

        rpg = rst_repo.rpgs[epoch%len(rst_repo.rpgs)]
        Q = np.where(train_idx)[0]
        pred_lp = utils.predict_fast(rpg, Q, data.y.cpu().detach().numpy(), max_dist=self.vargs['lp_max_dist'])
        pred_lp = np.array([-1 if v is None else v for v in pred_lp])
        mask = pred_lp>=0

        model.train()
        optimizer.zero_grad()
        F.nll_loss(model(data.x, edge_index_rst)[mask], torch.tensor(pred_lp).to(device)[mask]).backward()
        optimizer.step()

    def train_slow(self, epoch, model, optimizer, data, train_idx):
        rpg_index = epoch%len(rst_repo.rpgs)
        edge_index_rpg = rst_repo.get_edge_index(rpg_index)

        model.train()
        optimizer.zero_grad()
        F.nll_loss(model(data.x, edge_index_rpg)[train_idx], data.y[train_idx]).backward()
        optimizer.step()

class RST_LP2(PygExperiment):
    def train_step(self, epoch, model, optimizer, data, train_idx):
        rpg_index = epoch%len(rst_repo.rpgs)
        rpg = rst_repo.rpgs[rpg_index]
        edge_index_rpg = rst_repo.get_edge_index(rpg_index)

        Q = np.where(train_idx)[0]
        pred_lp = utils.predict_fast(rpg, Q, data.y.cpu().detach().numpy(), max_dist=self.vargs['lp_max_dist'])
        pred_lp = np.array([-1 if v is None else v for v in pred_lp])
        mask = pred_lp>=0

        model.train()
        optimizer.zero_grad()
        F.nll_loss(model(data.x, edge_index_rpg)[mask], torch.tensor(pred_lp).to(device)[mask]).backward()
        optimizer.step()

def predict_many(rpgs, Q, y, max_dist):
    """
    Create a prediction array for each rpg, for each node, using label propagation.
    Nodes for which the propagation surpasses max_dist from a labeled node
        are assigned with -1.
    Returns an array with shape (len(y), len(rpgs). 
    """
    preds_lp = []
    for rpg in rst_repo.rpgs:
        assert len(rpg) == len(y), f"rpg with {len(rpg)} nodes, y {len(y)}"
        pred_lp = utils.predict_fast(rpg, Q, y, max_dist=max_dist)
        pred_lp = np.array([-1 if v is None else v for v in pred_lp])
        preds_lp.append(pred_lp)

    return np.vstack(preds_lp).T

def aggregate_labels_most_common(pred_nodes):
    """
    Aggregate the predicted labels by selecting the most common.
    """
    # min_labeled = preds_lp.shape[1]*0.2
    min_labeled = 1
    def most_common(a):
        a = a[a>=0]
        return np.bincount(a).argmax() if len(a) >= min_labeled else -1
    preds = np.array([most_common(col) for col in pred_nodes])
    return preds

def aggregate_labels_weighted(preds_lp):
    """
    Aggregate the predicted labels by selecting the most common
    and returning a weight associated to it.
    """
    def weighted_most_common(a):
        tot = a.shape[0]
        a = a[a>=0]
        if len(a) == 0:
            return -1, 1.
        counts = np.bincount(a)
        most_common = counts.argmax()
        return most_common, counts[most_common] / tot

    weighted_labels = [weighted_most_common(row) for row in preds_lp]
    labels = np.array([l for l, w in weighted_labels])
    weights = np.array([w for l, w in weighted_labels])
    return labels, weights

def predict_many_ohe(rpgs, Q, y_ohe):
    """
    Create a prediction array for each rpg, for each node, using label propagation.
    Nodes for which the propagation surpasses max_dist from a labeled node
        are assigned with -1.
    Returns an array with shape (len(y), len(rpgs). 
    """

    scores = np.zeros_like(y_ohe)
    for rpg in rpgs:
        dist = utils.propagate_ohe(rpg, Q, y_ohe)
        dist[dist <= 1] = 0
        score = np.exp(-dist)
        # score[dist > 1] = 0
        scores += score
    pred = scores.argmax(-1)
    # pred[scores[np.arange(scores.shape[0]), pred] < len(rpgs)*np.exp(-2)] = -1
    pred[scores[np.arange(scores.shape[0]), pred] == 0] = -1
    # print(pred.shape[0], (pred>=0).sum(), Q.shape[0])
    return pred
    
class LP_Precomputed_Experiment(PygExperiment):
    
    def train_step_args(self):
        lp_max_dist = self.vargs["lp_max_dist"]
        y = to_numpy(self.data.y)
        Q = self.train_idx

        if lp_max_dist is None or lp_max_dist < 0:
            y_ohe = np.zeros((y.shape[0], self.data.num_classes))
            y_ohe[np.arange(y.shape[0]), y.astype(int)] = 1

            pred_lp = predict_many_ohe(rst_repo.rpgs, Q, y_ohe)
            return {"pred_lp": pred_lp}

        preds_lp = predict_many(rst_repo.rpgs, Q, y, lp_max_dist)
        #return {"pred_lp": preds_lp}
        
        # # unweighted most common
        # pred_lp = aggregate_labels_most_common(preds_lp)
        # return {"pred_lp": pred_lp}
    
        # weighted most common
        pred_lp, weights = aggregate_labels_weighted(preds_lp)
        return {"pred_lp": pred_lp, "pred_weights": weights}

def one_choice_per_row(a):
    """
    Choose one value at a random position for each row of a.
    Returns a vector with length a.shape[0]
    """
    c = np.random.choice(a.shape[1], size=a.shape[0])
    return a[np.arange(a.shape[0]), c]

def train_step_common(model, optimizer, x, edge_index, y, mask, weights=None):
    model.train()
    optimizer.zero_grad()
    losses = F.nll_loss(model(x, edge_index)[mask], y[mask], reduction='none')
    if weights is None:
        loss = losses.mean()
    else:
        loss = (losses * torch.tensor(weights[mask]).to(device)).mean()
    loss.backward()
    optimizer.step()

class RPG_LP(LP_Precomputed_Experiment):

    def train_step(self, epoch, model, optimizer, data, train_idx, pred_lp, pred_weights=None):
        rpg_index = epoch%len(rst_repo.rpgs)
        edge_index_rpg = rst_repo.get_edge_index(rpg_index)
        
        #pred_lp = one_choice_per_row(pred_lp)
        mask = pred_lp>=0
        
        train_step_common(model, optimizer, data.x, edge_index_rpg,
                y=torch.tensor(pred_lp).to(device), mask=mask, weights=pred_weights)

class FULL_LP(LP_Precomputed_Experiment):
    def train_step(self, epoch, model, optimizer, data, train_idx, pred_lp, pred_weights=None):
        #pred_lp = one_choice_per_row(pred_lp)
        mask = pred_lp>=0
        
        train_step_common(model, optimizer, data.x, data.edge_index,
                y=torch.tensor(pred_lp).to(device), mask=mask, weights=pred_weights)


class RST_LP(LP_Precomputed_Experiment):

    def train_step(self, epoch, model, optimizer, data, train_idx, pred_lp, pred_weights=None):
        rst = rst_repo.rsts[epoch%len(rst_repo.rsts)]
        edge_index_rst = torch.concat([torch.tensor(list(rst.edges)), 
                                    torch.tensor([(t,s) for (s,t) in rst.edges])]).T.to(device)
        
        #pred_lp = one_choice_per_row(pred_lp)
        mask = pred_lp>=0
        
        train_step_common(model, optimizer, data.x, edge_index_rst,
                y=torch.tensor(pred_lp).to(device), mask=mask, weights=pred_weights)

def poison_data(data, nr_nodes, seed=0):
    rng = np.random.default_rng(seed)
    data = data.clone()
    degree = tgu.degree(data.edge_index[0]) + tgu.degree(data.edge_index[1])
    degree_sorted = degree.argsort(descending=True)[:nr_nodes]
    for n in degree_sorted:
        for i in range(10000):
            picked = rng.integers(data.num_nodes)
            if data.y[picked] != data.y[n]:
                break
        else:
            picked = rng.integers(data.num_nodes)
        x_picked = data.x[picked].clone()
        data.x[picked] = data.x[n]
        data.x[n] = x_picked
    return data


from torch_geometric.loader import GraphSAINTRandomWalkSampler

class GRAPH_SAINT(PygExperiment):
    
    def __init__(self, use_normalization=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_normalization = False
        self.data = self.data.to('cpu')
        self.data.node_index = torch.arange(self.data.num_nodes)
        self.train_loader = GraphSAINTRandomWalkSampler(self.data, batch_size=6000, walk_length=2,
                                     num_steps=5, sample_coverage=100,
                                     save_dir=f"./dataset/{self.data.dataset_name.replace('-', '_')}/processed/",
                                     num_workers=4)
        
    def train_step_args(self):
        self.train_mask = to_tensor(self.train_mask)
        self.data = self.data.to('cpu')
        
        return super().train_step_args()

    def train_step(self, epoch, model, optimizer, data, train_idx):
        data.to('cpu')

        model.train()
        model.set_aggr('add' if self.use_normalization else 'mean')

        total_loss = total_examples = 0
        for data in self.train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            assert data.x is not None

            if self.use_normalization:
                edge_weight = data.edge_norm * data.edge_weight
                out = model(data.x, data.edge_index, edge_weight)
                loss = F.nll_loss(out, data.y, reduction='none')
                loss = (loss * data.node_norm)[data.train_mask].sum()
            else:
                out = model(data.x, data.edge_index)
                loss = F.nll_loss(out[self.train_mask[data.node_index]], data.y[self.train_mask[data.node_index]])

            loss.backward()
            optimizer.step()
            
    def test_step(self, model, data, *masks):
        data = data.to(device)
        return super().test_step(model, data, *masks)
    
class SAGE_EXP(PygExperiment):

    def __init__(self, data, split, verbose=0, **vargs):
        super().__init__(data, split, verbose, **vargs)

        self.batch_inference = True

    def setup(self, model, optimizer, seed):
        super().setup(model, optimizer, seed)

        self.data = self.data.to('cpu')
        self.data = self.data.to(device, 'x', 'y')

        kwargs = {'batch_size': 1024, 'num_workers': 4, 'persistent_workers': True}
        # kwargs = {'batch_size': 1024, 'num_workers': 0}
        self.train_loader = NeighborLoader(copy.copy(self.data),
                input_nodes=to_tensor(self.train_mask),
                num_neighbors=[25, 10], shuffle=True, **kwargs)
    
    def train_step(self, epoch, model, optimizer, data, train_idx):

        model.train()

        for batch in self.train_loader:
            
            optimizer.zero_grad()
            y = batch.y[:batch.batch_size]
            y_hat = model(batch.x, batch.edge_index.to(device))[:batch.batch_size]
            loss = F.nll_loss(y_hat, y)
            loss.backward()
            optimizer.step()
