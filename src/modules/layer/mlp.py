import torch.nn as nn
import torch
import numpy as np

class MLP(nn.Module):

	def __init__(self, input_dim, output_dim, layer_sizes, batchnorm=False, layernorm=False, nonlinearity=nn.ReLU):
		super(MLP, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.batchnorm = batchnorm
		layer_sizes_full = [input_dim] + layer_sizes + [output_dim]
		layers = []
		for i in range(len(layer_sizes_full)-1):
			layers.append(nn.Linear(layer_sizes_full[i], layer_sizes_full[i+1]))
			if i != len(layer_sizes_full)-2:
				if batchnorm:
					layers.append(BatchNorm(layer_sizes_full[i+1]))
				if layernorm:
					layers.append(nn.LayerNorm(layer_sizes_full[i+1]))
				layers.append(nonlinearity())
		self.net = nn.Sequential(*layers)

	def forward(self, X):
		return self.net(X)



class MultiModule(nn.Module):

	def __init__(self, n_agents, module=MLP, *args, **kwargs):
		super().__init__()
		self.n_agents = n_agents
		self.mlps = nn.ModuleList([module(*args, **kwargs) for _ in range(self.n_agents)])
		self.mlp_modules = list(self.mlps.children())

	def forward(self, X):
		# B1, ..., Bn, n_agents, input_dim
		return torch.stack([self.mlp_modules[i](select_index(X, -2, i)) for i in range(self.n_agents)], dim=-2)



class BatchNorm(nn.Module):

	def __init__(self, *args, **kwargs):
		super().__init__()
		self.bn = nn.BatchNorm1d(*args, **kwargs)

	def forward(self, x):
		shape = x.shape
		x_r = x.reshape(np.prod(shape[:-1]), shape[-1])
		y_r = self.bn(x_r)
		y = y_r.reshape(shape)
		return y


def layers(input_dim, output_dim, nlayers=1, midmult=1):
	midlayersize = midmult * (input_dim + output_dim)//2
	midlayersize = max(midlayersize, 1)
	nlayers += 2
	layers1 = np.around(np.logspace(np.log10(input_dim), np.log10(midlayersize), num=(nlayers)//2)).astype(int)
	layers2 = np.around(np.logspace(np.log10(midlayersize), np.log10(output_dim), num=(nlayers+1)//2)).astype(int)[1:]
	return list(np.concatenate([layers1, layers2])[1:-1])


def build_mlp(input_dim, output_dim, nlayers=1, midmult=1., batchnorm=False, layernorm=False, nonlinearity=nn.ReLU):
	mlp_layers = layers(input_dim=input_dim, output_dim=output_dim, nlayers=nlayers, midmult=midmult)
	mlp = MLP(input_dim=input_dim, output_dim=output_dim, layer_sizes=layers, batchnorm=batchnorm, layernorm=layernorm, nonlinearity=nonlinearity)
	return mlp


def select_index(arr, dim, idx):
	idx_list = [slice(None)] * len(arr.shape)
	idx_list[dim] = idx
	return arr.__getitem__(idx_list)


if __name__ == '__main__':
	f = MultiModule(8, MLP, input_dim=3, output_dim=5, layer_sizes=[4])
	x = torch.randn(10, 8, 3)
	y = f(x)
	print(y.shape)