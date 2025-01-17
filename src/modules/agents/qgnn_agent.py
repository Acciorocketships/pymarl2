import torch
from torch import nn
import numpy as np
from torch_geometric.nn import Sequential, EdgeConv, GraphConv
from pymarl.modules.layer.agggnn import AggGNN, create_agg_gnn
from pymarl.modules.layer.mlp import MLP
from pymarl.modules.layer.gnn_wrapper import GNNwrapper
from torch_geometric.data import Data
from torch_geometric import utils



class RNN(nn.Module):
	def __init__(self, input_shape, args):
		super(RNN, self).__init__()
		self.args = args

		self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
		self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

		if getattr(args, "model_use_layernorm", False):
			self.layer_norm = nn.LayerNorm(args.rnn_hidden_dim)


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
		super().__init__()
		self.hidden_dim = args.rnn_hidden_dim
		self.out_dim = args.n_actions
		self.use_layernorm = getattr(args, "model_use_layernorm", False)
		self.adj_dropout = getattr(args, "model_adj_dropout", 0.)
		self.model_gnn_type = getattr(args, "model_gnn_type", "edgeconv")
		self.model_gnn_layers = getattr(args, "model_gnn_layers", 1)

		# Trajectory Encoder
		self.rnn = RNN(input_shape, args)

		# GNNs
		self.gnn = gnn_builder(gnn_type=self.model_gnn_type, layers=self.model_gnn_layers, dim=self.hidden_dim, layernorm=self.use_layernorm)
		# Q Net
		self.q_net = MLP(input_dim=2 * self.hidden_dim, output_dim=self.out_dim, layer_sizes=[(input_shape + self.hidden_dim + self.out_dim) // 2], layernorm=self.use_layernorm)


	def forward(self, inputs, hidden_state, info={}):
		batch, n_agents, obs_dim = inputs.size()

		h = self.rnn(inputs, hidden_state)

		adj_raw = info.get('adj', None)
		adj = self.get_adj(adj_raw, batch, n_agents)
		gnn_out = self.gnn(h, adj)

		embedding = torch.cat([h, gnn_out], dim=-1)

		qvals = self.q_net(embedding)

		return qvals, h



	def get_adj(self, adj, batch, n_agents):
		if isinstance(adj, Data):
			adj.edge_index = utils.add_self_loops(adj.edge_index)[0]
			if self.adj_dropout != 0:
				adj.edge_index = utils.dropout_adj(adj.edge_index, p=self.adj_dropout)[0]
			return adj
		else:
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
			if isinstance(adj, torch.Tensor):
				adj = adj.cpu().numpy()
			diag = adj.diagonal(axis1=-2, axis2=-1)
			diag.setflags(write=True)
			diag.fill(1)
			return adj



	def init_hidden(self):
		return self.rnn.init_hidden()



def gnn_builder(gnn_type='edgeconv', layers=1, dim=64, layernorm=False, **kwargs):
	
	gnn_layers = []

	if gnn_type == 'edgeconv':
		for i in range(layers):
			net = MLP(input_dim=2*dim, output_dim=dim, layer_sizes=[dim*3//2], layernorm=layernorm)
			edgeconv_layer = EdgeConv(nn=net, aggr='mean')
			gnn_layers.append((edgeconv_layer, 'x, edge_index -> x'))
			if i+1 < layers:
				gnn_layers.append(nn.ReLU(inplace=True))

	elif gnn_type == 'graphconv':
		for i in range(layers):
			graphconv_layer = GraphConv(in_channels=dim, out_channels=dim, aggr='mean')
			gnn_layers.append((graphconv_layer, 'x, edge_index -> x'))
			if i+1 < layers:
				gnn_layers.append(nn.ReLU(inplace=True))

	elif gnn_type == 'agggnn':
		for i in range(layers):
			agggnn_layer = create_agg_gnn(in_dim=dim, out_dim=dim, nlayers=2, midmult=1., layernorm=layernorm, fcom=None,  aggr='mean')
			gnn_layers.append((agggnn_layer, 'x, edge_index -> x'))
			if i+1 < layers:
				gnn_layers.append(nn.ReLU(inplace=True))

	elif gnn_type == 'mha':
		from pymarl.modules.layer.gnns import MHAconv
		for i in range(layers):
			heads = 2
			mha_layer = MHAconv(input_size=dim, heads=heads, embed_size=dim//heads)
			gnn_layers.append((mha_layer, 'x, edge_index -> x'))
			if i + 1 < layers:
				gnn_layers.append(nn.ReLU(inplace=True))
	
	gnn_geometric = Sequential('x, edge_index', gnn_layers)
	gnn = GNNwrapper(gnn_geometric)
	return gnn







