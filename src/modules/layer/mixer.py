from torch import nn
import torch
from modules.layer.mlp import MLP, MultiModule, layers
from modules.layer.genagg import GenAgg


class Mixer(nn.Module):

	def __init__(self, input_dim, output_dim, hidden_dim, psi_layers=2, phi_layers=2, layernorm=False, midmult=1., heterogeneous=False, n_agents=None):
		super().__init__()
		self.psi_layers = psi_layers
		self.phi_layers = phi_layers
		if self.psi_layers > 0:
			psi_layer_sizes = layers(input_dim=input_dim, output_dim=hidden_dim, nlayers=psi_layers, midmult=midmult)
			if not heterogeneous:
				self.psi = MLP(input_dim=input_dim, output_dim=hidden_dim, layer_sizes=psi_layer_sizes, layernorm=layernorm)
			else:
				self.psi = MultiModule(n_agents=n_agents, module=MLP, input_dim=input_dim, output_dim=hidden_dim, layer_sizes=psi_layer_sizes, layernorm=layernorm)
		if self.phi_layers > 0:
			phi_layer_sizes = layers(input_dim=hidden_dim, output_dim=output_dim, nlayers=phi_layers, midmult=midmult)
			if not heterogeneous:
				self.phi = MLP(input_dim=hidden_dim, output_dim=output_dim, layer_sizes=phi_layer_sizes, layernorm=layernorm)
			else:
				self.phi = MultiModule(n_agents=n_agents, module=MLP, input_dim=hidden_dim, output_dim=output_dim, layer_sizes=phi_layer_sizes, layernorm=layernorm)


	def forward(self, X):
		# X: batch x N x input_dim
		# output: batch x output_dim
		if self.psi_layers > 0:
			local_embed = self.psi(X)
		else:
			local_embed = X
		local_embed_sum = local_embed.sum(dim=-2)
		if self.phi_layers > 0:
			global_embed = self.phi(local_embed_sum)
		else:
			global_embed = local_embed_sum
		return global_embed



class AggMixer(nn.Module):

	def __init__(self, input_dim, output_dim, hidden_dim, psi_layers=2, phi_layers=2, layernorm=False, midmult=1., heterogeneous=False, n_agents=None):
		super().__init__()
		self.psi_layers = psi_layers
		self.phi_layers = phi_layers
		self.agg = GenAgg(p=1., a=0., shift=True, learnable=True)
		if self.psi_layers > 0:
			psi_layer_sizes = layers(input_dim=input_dim, output_dim=hidden_dim, nlayers=psi_layers, midmult=midmult)
			if not heterogeneous:
				self.psi = MLP(input_dim=input_dim, output_dim=hidden_dim, layer_sizes=psi_layer_sizes, layernorm=layernorm, nonlinearity=nn.PReLU)
			else:
				self.psi = MultiModule(n_agents=n_agents, module=MLP, input_dim=input_dim, output_dim=hidden_dim, layer_sizes=psi_layer_sizes, layernorm=layernorm, nonlinearity=nn.PReLU)
		if self.phi_layers > 0:
			phi_layer_sizes = layers(input_dim=hidden_dim, output_dim=output_dim, nlayers=phi_layers, midmult=midmult)
			if not heterogeneous:
				self.phi = MLP(input_dim=hidden_dim, output_dim=output_dim, layer_sizes=phi_layer_sizes, layernorm=layernorm, nonlinearity=nn.PReLU)
			else:
				self.phi = MultiModule(n_agents=n_agents, module=MLP, input_dim=hidden_dim, output_dim=output_dim, layer_sizes=phi_layer_sizes, layernorm=layernorm, nonlinearity=nn.PReLU)


	def forward(self, X):
		# X: batch x N x input_dim
		# output: batch x output_dim
		if self.psi_layers > 0:
			local_embed = self.psi(X)
		else:
			local_embed = X
		local_embed_sum = self.agg(local_embed)
		if self.phi_layers > 0:
			global_embed = self.phi(local_embed_sum)
		else:
			global_embed = local_embed_sum
		return global_embed