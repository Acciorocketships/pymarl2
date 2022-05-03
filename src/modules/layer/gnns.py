import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax

from torch import Tensor
from typing import Optional

class MHAconv(MessagePassing):

    def __init__(self, input_size, heads, embed_size):
        super().__init__()
        self.input_size = input_size
        self.heads = heads
        self.emb_size = embed_size
        self.aggr = 'sum'

        self.tokeys = nn.Linear(self.input_size, self.emb_size * self.heads, bias = False)
        self.toqueries = nn.Linear(self.input_size, self.emb_size * self.heads, bias = False)
        self.tovalues = nn.Linear(self.input_size, self.emb_size * self.heads, bias = False)

    def forward(self, x: Tensor, edge_index: Tensor):
        # x: batch*n_agents x  input_size
        n_agents, input_size = x.size()

        keys = self.tokeys(x)
        queries = self.toqueries(x)
        values = self.tovalues(x)

        queries = queries / (self.emb_size ** (1 / 4))
        keys = keys / (self.emb_size ** (1 / 4))

        weights = self.edge_updater(edge_index=edge_index, keys=keys, queries=queries, adj=edge_index)

        agg = self.propagate(edge_index=edge_index, x=x, weights=weights, values=values)

        return agg.view(n_agents, self.heads * self.emb_size)


    def message(self, values_j: Tensor, weights: Tensor) -> Tensor:
        n_edges, _ = values_j.shape
        values_j = values_j.view(n_edges, self.heads, self.emb_size)
        out = values_j * weights[:,:,None]
        return out

    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        n_edges, heads, emb_size = inputs.shape
        out = scatter(inputs.view(n_edges, self.heads * self.emb_size), index, dim=self.node_dim, dim_size=dim_size, reduce='sum')
        return out.view(out.shape[0], self.heads, self.emb_size)

    def edge_update(self, keys_j : Tensor, queries_i : Tensor, adj: Tensor) -> Tensor:
        n_edges, _ = keys_j.shape
        keys_j = keys_j.view(n_edges, self.heads, self.emb_size)
        queries_i = queries_i.view(n_edges, self.heads, self.emb_size)
        dot = (keys_j * queries_i).sum(dim=-1)
        weights = scatter_softmax(src=dot, index=adj[1,:], dim=0)
        return weights

    def forward_batch(self, x, adj=None):
        b, t, hin = x.size()
        assert hin == self.input_size, f'Input size {{hin}} should match {{self.input_size}}'

        h = self.heads
        e = self.emb_size

        keys = self.tokeys(x).view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values = self.tovalues(x).view(b, t, h, e)

        # dot-product attention
        # folding heads to batch dimensions
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))

        dot = torch.bmm(queries, keys.transpose(1, 2))
        assert dot.size() == (b * h, t, t)

        if adj is not None:
            adj_exp = adj.unsqueeze(1).repeat((1, h, 1, 1)).view(b * h, t, t).transpose(1,2)
            dot[adj_exp == 0] = -float('inf')

        # row wise self attention probabilities
        dot = nn.functional.softmax(dot, dim=2)
        out = torch.bmm(dot, values).view(b, h, t, e)
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)
        out[torch.isnan(out)] = 0
        return out




## TESTS ##

def test1():
    from pymarl.modules.layer.gnn_wrapper import GNNwrapper

    batch_size = 1
    n_agents = 5
    input_size = 4
    heads = 2
    embed_size = 3

    x = torch.randn(batch_size, n_agents, input_size)
    A = torch.rand(batch_size, n_agents, n_agents) < 0.3

    gnn = MHAconv(input_size=input_size, heads=heads, embed_size=embed_size)
    dense_gnn = GNNwrapper(gnn)

    out_truth = gnn.forward_batch(x=x, adj=A)
    out_gnn = dense_gnn.forward(X=x, A=A)

    print(out_gnn, out_truth)
    print(torch.norm(out_truth - out_gnn))

def test2():
    from pymarl.modules.layer.gnn_wrapper import GNNwrapper

    batch_size = 1
    n_agents = 5
    input_size = 4
    heads = 2
    embed_size = 3

    x = torch.randn(batch_size, n_agents, input_size)
    A = torch.eye(n_agents).repeat((batch_size, 1, 1))
    A[:,0,1] = 1

    gnn = MHAconv(input_size=input_size, heads=heads, embed_size=embed_size)
    dense_gnn = GNNwrapper(gnn)

    out_gnn = dense_gnn.forward(X=x, A=A)

    print(out_gnn)
    # if a_ij = 1 then agent i receives data from agent j


if __name__ == "__main__":
    test1()
