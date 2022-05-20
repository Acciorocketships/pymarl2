import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix


class GNNwrapper(nn.Module):

	def __init__(self, gnn, input_func = lambda data: {"edge_index": data.edge_index, "x": data.x}):
		super().__init__()
		self.gnn = gnn
		self.input_func = input_func


	def forward(self, X, A):
		B, n_agents, _ = X.shape
		if isinstance(A, Batch):
			data = A
			data.x = X.reshape(B*n_agents, -1)
		else:
			data = to_geometric(A, x=X).to(X.device)
		output = self.gnn(**self.input_func(data))

		return output.reshape(B, n_agents, -1)



def to_geometric(A, **kwargs):
	A = torch.as_tensor(A).cpu()
	for name, val in kwargs.items():
		kwargs[name] = torch.as_tensor(val)
	numdim = len(A.shape)
	numbatchdim = numdim-2
	batchdimsize = int(np.prod(A.shape[:numbatchdim]))
	A = A.reshape(batchdimsize, *A.shape[numbatchdim:])
	for name, val in kwargs.items():
		kwargs[name] = val.reshape(batchdimsize, *val.shape[numbatchdim:])
	return dense_to_geometric(A=A, **kwargs)



def dense_to_geometric(A, **kwargs):
	edge_list = [from_scipy_sparse_matrix(coo_matrix(A[i])) for i in range(A.shape[0])]
	arg_slice = lambda i: {key: val[i] for key, val in kwargs.items()}
	data_list = [Data(edge_index=edge_index, edge_attr=edge_attr, **(arg_slice(i))) for i, (edge_index, edge_attr) in enumerate(edge_list)]
	return Batch.from_data_list(data_list)
