import torch
import torch.nn.functional as F
import torch_geometric as tg
from torch_geometric.nn.dense.linear import Linear
from tqdm import tqdm

class SAGEConv(tg.nn.MessagePassing):

    def __init__(
        self,
        in_channels,
        out_channels,
        aggr="mean"):
        
        super().__init__(aggr)
    
        self.lin_l = Linear(in_channels, out_channels, bias=True)
        self.lin_r = Linear(in_channels, out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        
    def forward(self, x, edge_index, size=None):
        out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l(out)
        out = out + self.lin_r(x)
        return out
    

class SAGE(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, 
                 out_channels, num_layers, dropout, use_bn=False):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(num_features, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout
        
        self.use_bn = use_bn
        if self.use_bn:
            self.bns = torch.nn.ModuleList()
            for _ in range(num_layers - 1):
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

    def set_aggr(self, aggr):
        for conv in self.convs:
            conv.aggr = aggr

    # def forward(self, x, edge_index):
    #     x = F.dropout(x, p=self.dropout, training=self.training)
    #     x = F.elu(self.conv1(x, edge_index))
    #     x = F.dropout(x, p=self.dropout, training=self.training)
    #     x = self.conv2(x, edge_index)
    #     return F.log_softmax(x, dim=1)
    
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]): # type: ignore
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader, verbose=True):
        device = self.convs[0].lin_l.bias.device
        if verbose:
            pbar = tqdm(total=len(subgraph_loader.dataset) * len(self.convs), position=0, leave=True)
            pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                if i < len(self.convs) - 1:
                    x = F.elu(x)
                xs.append(x[:batch.batch_size].cpu())
                if verbose:
                    pbar.update(batch.batch_size)
            x_all = torch.cat(xs, dim=0)
        if verbose:
            pbar.close()
        # return x_all
        return F.log_softmax(x_all, dim=1)


from torch_geometric.nn import SAGEConv as SAGEConvOrig, GCNConv

class SAGE_OGB(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, use_bn=True):
        super(SAGE_OGB, self).__init__()
        
        self.dropout = dropout
        self.use_bn = use_bn

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConvOrig(in_channels, hidden_channels))
        if self.use_bn:
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConvOrig(hidden_channels, hidden_channels))
            if self.use_bn:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConvOrig(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.use_bn:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x, adj_t, edge_weight=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t, edge_weight)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)
    
    @torch.no_grad()
    def inference(self, x_all, subgraph_loader, verbose=True):
        device = self.convs[0].lin_l.bias.device
        if verbose:
            pbar = tqdm(total=len(subgraph_loader.dataset) * len(self.convs), position=0, leave=True)
            pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                if i < len(self.convs) - 1:
                    if self.use_bn:
                        x = self.bns[i](x)
                    x = F.relu(x)
                xs.append(x[:batch.batch_size].cpu())
                if verbose:
                    pbar.update(batch.batch_size)
            x_all = torch.cat(xs, dim=0)
        if verbose:
            pbar.close()
        # return x_all
        return F.log_softmax(x_all, dim=1)
    
    def set_aggr(self, aggr):
        for conv in self.convs:
            conv.aggr = aggr
    
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, use_bn=False, cached=False):
        super(GCN, self).__init__()
        
        self.dropout = dropout
        self.use_bn = use_bn

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=cached))
        if self.use_bn:
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=cached))
            if self.use_bn:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=cached))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t, edge_weight=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t, edge_weight)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)
    
    def set_aggr(self, aggr):
        for conv in self.convs:
            conv.aggr = aggr
            
    @torch.no_grad()
    def inference(self, x_all, subgraph_loader, verbose=True):
        device = self.convs[0].lin.weight.device
        if verbose:
            pbar = tqdm(total=len(subgraph_loader.dataset) * len(self.convs), position=0, leave=True)
            pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                if i < len(self.convs) - 1:
                    if self.use_bn:
                        x = self.bns[i](x)
                    x = F.relu(x)
                xs.append(x[:batch.batch_size].cpu())
                if verbose:
                    pbar.update(batch.batch_size)
            x_all = torch.cat(xs, dim=0)
        if verbose:
            pbar.close()
        # return x_all
        return F.log_softmax(x_all, dim=1)