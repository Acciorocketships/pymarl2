import torch
import torch.nn as nn
from torch_geometric.nn import Sequential, EdgeConv
from pymarl.modules.layer.mlp import MLP
from pymarl.modules.layer.gnn_wrapper import GNNwrapper
from pymarl.modules.layer.mixer import Mixer


class ComaGNNCritic(nn.Module):
    def __init__(self, scheme, args):
        super(ComaGNNCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        self.input_shape = self._get_input_shape(scheme)
        self.output_type = "q"

        # Set up network layers
        self.gnn_layers = [32, 32, 32]
        self.gnn = GNNwrapper(self.build_gnn())
        self.pooling = Mixer(input_dim=self.gnn_layers[-1], output_dim=1, hidden_dim=self.gnn_layers[-1]//2,
                             psi_layers=2, phi_layers=2, batchnorm=False, midmult=1.)

    def forward(self, inputs):
        # B x T x N x D
        b, t, n, d = inputs.shape
        inputs_reshaped = inputs.view(b*t, n, d)
        adj = torch.ones(b*t, n, n)
        x = self.gnn(X=inputs_reshaped, A=adj)
        y = self.pooling(x)
        out = y.view(b, t, 1)
        return out # B x T x 1



    def _get_input_shape(self, scheme):
        # observation and action
        return scheme["obs"]["vshape"] + scheme["actions_onehot"]["vshape"][0]


    def build_gnn(self):
        gnn_layers = []
        l = 1
        for in_dim, out_dim in zip([self.input_shape] + self.gnn_layers[:-1], self.gnn_layers):
            net = MLP(input_dim=2 * in_dim, output_dim=out_dim, layer_sizes=[(in_dim + out_dim) // 2], layernorm=True)
            edgeconv_layer = EdgeConv(nn=net, aggr='mean')
            gnn_layers.append((edgeconv_layer, 'x, edge_index -> x'))
            if l < len(self.gnn_layers):
                gnn_layers.append(nn.ReLU(inplace=True))
            l += 1
        gnn = Sequential('x, edge_index', gnn_layers)
        return gnn