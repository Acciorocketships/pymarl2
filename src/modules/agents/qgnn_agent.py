from torch import nn
import torch
import numpy as np
# from utils.th_utils import orthogonal_init_
from torch_geometric.nn import EdgeConv
from modules.layer.mlp import MLP
from modules.layer.gnn_wrapper import GNNwrapper



class RNN(nn.Module):
    def __init__(self, input_shape, args):
        super(RNN, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        if getattr(args, "model_use_layernorm", False):
            self.layer_norm = nn.LayerNorm(args.rnn_hidden_dim)
        
        # if getattr(args, "use_orthogonal", False):
        #     orthogonal_init_(self.fc1)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        b, a, e = inputs.size()

        inputs = inputs.view(-1, e)
        x = nn.functional.relu(self.fc1(inputs), inplace=True)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        hh = self.rnn(x, h_in)

        return hh.view(b, a, -1)



class QGNNAgent(nn.Module):

    def __init__(self, input_shape, args):
        nn.Module.__init__(self)
        self.hidden_dim = args.rnn_hidden_dim
        self.out_dim = args.n_actions
        self.use_layernorm = getattr(args, "model_use_layernorm", False)
        self.adj_dropout = getattr(args, "adj_dropout", 0.)

        # Trajectory Encoder
        self.rnn = RNN(input_shape, args)

        # GNNs
        self.edgeconv_nn_actor = MLP(input_dim=2*self.hidden_dim, output_dim=self.hidden_dim, layer_sizes=[self.hidden_dim*3//2], batchnorm=self.use_layernorm)
        self.gnn_geometric = EdgeConv(nn=self.edgeconv_nn_actor, aggr='mean')
        self.gnn = GNNwrapper(self.gnn_geometric)

        # Q Net
        self.q_net = MLP(input_dim=self.hidden_dim, output_dim=self.out_dim, layer_sizes=[(self.hidden_dim+self.out_dim)//2], batchnorm=self.use_layernorm)

        

    def forward(self, inputs, hidden_state, adj=None):
        batch, n_agents, obs_dim = inputs.size()

        h = self.rnn(inputs, hidden_state)

        adj = self.get_adj(adj, batch, n_agents)
        embedding = self.gnn(h, adj)

        qvals = self.q_net(embedding)

        return qvals, h



    def get_adj(self, adj, batch, n_agents, device):
        if adj is not None:
            adj = adj.reshape(batch, n_agents, n_agents)
        else:
            adj = np.ones((batch, n_agents, n_agents))
        if self.adj_dropout != 0:
            if self.adj_dropout == 1:
                adj = np.eye(n_agents)[np.newaxis].repeat(batch, axis=0)
            elif self.adj_dropout == None:
                adj = np.zeros(batch, n_agents, n_agents)
            else:
                edges = adj.nonzero()
                n_edges = len(edges[0])
                mask = np.random.rand(n_edges) < self.adj_dropout
                dropped_edges = tuple([dim[mask] for dim in edges])
                adj[dropped_edges] = 0
                diag = adj.diagonal(axis1=-2, axis2=-1)
                diag.setflags(write=True)
                diag.fill(1)
        return adj



    def init_hidden(self):
        return self.rnn.init_hidden()



