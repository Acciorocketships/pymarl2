import torch
import math
from typing import Callable, Tuple
from torch_scatter import scatter




class GenAgg(torch.nn.Module):
	def __init__(self,
			# p term in generalized mean
			p: float = 1.0, 
			# premultiply gen-mean by N^a 
			# so we can normalize wrt to num neighbors
			a: float = 0.0, 
			# Whether to shift x into positive real to avoid
			# complex numbers
			shift: bool = False, 
			# Whether a,p should be learnable or fixed
			learnable: bool = True, 
			# Mapping function from parameter p' => actual p
			# use this if we don't want p to change linearly
			# (e.g. we want to approach p=inf faster, so map 
			# p = tan(p'))
			map_fn: Callable = lambda p: torch.tan(p * math.pi / 4)
	):
		super().__init__()
		self.p = torch.nn.Parameter(
			torch.tensor([p]).float()
		) if learnable else torch.tensor([p]).float()
		self.a = torch.nn.Parameter(
			torch.tensor([a]).float()
		) if learnable else torch.tensor([a]).float()
		self.shift = shift
		self.learnable = learnable
		self.p_domain = [-1.99, 1.99]
		self.map_fn = map_fn

	def forward(self, x):
		# Clamp p parameter to prevent over/underflows
		if self.learnable:
			self.p.data = self.p.data.clamp(*self.p_domain)
		if self.map_fn:
			p = self.map_fn(self.p)

		y = gen_agg(x, p=p, a=self.a, shift=self.shift, special=(not self.learnable))
		if torch.is_complex(y):
			y = y.real
		return y

	
def gen_agg(x, p=1, a=0, agg_dim=-2, shift=False, special=True):
	"""Generalized mean aggregation. If shift is true,
	then apply transformations to ensure well-behaved outputs.
	When calling this method, ensure p is bounded to a reasonable
	value to prevent overflows"""

	# special cases
	if special:
		if p == math.inf:
			if torch.is_complex(x):
				x = x.real
			return torch.max(x, dim=agg_dim)[0]
		elif p == -math.inf:
			if torch.is_complex(x):
				x = x.real
			return torch.min(x, dim=agg_dim)[0]
		elif p == 0:
			dtype = x.dtype
			if torch.is_complex(x):
				x = x.real
			prod = torch.prod(x, dim=agg_dim)
			Y = torch.pow(prod.cfloat(), 1/x.shape[agg_dim])
			return Y.type(dtype)

	# number of neighbours
	N = torch.tensor(x.shape[agg_dim], device=x.device).float()

	# when shift is True, p=0 is min, p=1 is avg, p=inf is max
	if shift:
		if torch.is_complex(x):
			x = x.real
		shifts = x.min(dim=agg_dim)[0].unsqueeze(agg_dim)
		shifts -= 1e-3
		x = x - shifts
		Y = torch.exp(1 / p * (-torch.log(N) + logsumexp(p * torch.log(x), agg_dim=agg_dim)))
		res = N ** a * (Y + shifts)
		return res.select(dim=agg_dim, index=0)
	# when shift is False, p=-inf is min, p=0 is prod, p=1 is avg, p=inf is max
	else:
		res = torch.exp(a * torch.log(N) + (1 / p * (-torch.log(N) + logsumexp(p * torch.log(torch.abs(x)), agg_dim=agg_dim))))
		res = res.select(dim=agg_dim, index=0)
		return res




def logsumexp(x, agg_dim=-2):
	"""log-sum-exp that works for both real and complex
	torch tensors"""
	dtype = x.dtype
	if torch.is_complex(x):
		c_max, _ = x.real.max(dim=agg_dim)
		c_min, _ = x.real.min(dim=agg_dim)
		x = x.cdouble()
	else:
		c_max, _ = x.max(dim=agg_dim)
		c_min, _ = x.min(dim=agg_dim)
		x = x.double()
	c = c_max
	c = c.unsqueeze(agg_dim)
	shifted = torch.exp(x - c)
	summed = torch.sum(shifted, dim=agg_dim).unsqueeze(agg_dim)
	res = c + torch.log(summed).type(dtype)
	return res



class GenAggSparse(torch.nn.Module):
	"""Generalized aggregation operator"""
	def __init__(self,
			# p term in generalized mean
			p: float = 1.0, 
			# premultiply gen-mean by N^a 
			# so we can normalize wrt to num neighbors
			a: float = 0.0, 
			# Whether to shift x into positive real to avoid
			# complex numbers
			shift: bool = False, 
			# Whether a,p should be learnable or fixed
			learnable: bool = True, 
			# Mapping function from parameter p' => actual p
			# use this if we don't want p to change linearly
			# (e.g. we want to approach p=inf faster, so map 
			# p = tan(p'))
			map_fn: Callable = lambda p: torch.tan(p * math.pi / 4),
			# Bounds the domain of p to prevent over/underflow
			# for the default map_fn, this should be betwee -1.99.. and 1.99..
			p_domain: Tuple[float, float] = [-1.9999, 1.9999]
	):
		super().__init__()
		self.p = torch.nn.Parameter(torch.tensor([p]).float()) if learnable else torch.tensor([p]).float()
		self.a = torch.nn.Parameter(torch.tensor([a]).float()) if learnable else torch.tensor([a]).float()
		self.shift = shift
		self.learnable = learnable
		self.p_domain = p_domain
		self.map_fn = map_fn

	def forward(self, x, index, **kwargs):
		# Clamp p parameter to prevent over/underflows
		if self.learnable:
			self.p.data = self.p.data.clamp(*self.p_domain)
		if self.map_fn:
			p = self.map_fn(self.p)

		y = gen_agg_sparse(x, index, p=p, a=self.a, shift=self.shift, special=(not self.learnable), **kwargs)
		if torch.is_complex(y):
			y = y.real
		return y



def gen_agg_sparse(x, index, p=1, a=0, shift=False, special=False, agg_dim=0, dim_size=None):
	"""Generalized mean aggregation. If shift is true,
	then apply transformations to ensure well-behaved outputs."""

	if x.numel() == 0:
		shape = list(x.shape)
		shape[agg_dim] = dim_size
		return torch.zeros(shape, device=x.device, dtype=x.dtype)

	# special cases
	if special:
		if p == math.inf:
			if torch.is_complex(x):
				x = x.real
			return scatter(x, index, dim=agg_dim, dim_size=dim_size, reduce="max")
		elif p == -math.inf:
			if torch.is_complex(x):
				x = x.real
			return scatter(x, index, dim=agg_dim, dim_size=dim_size, reduce="min")
		elif p == 0:
			dtype = x.dtype
			if torch.is_complex(x):
				x = x.real
			N = scatter(torch.ones(x.shape[agg_dim], device=x.device), index, dim=agg_dim, dim_size=dim_size, reduce="sum")
			prod = scatter(x, index, dim=agg_dim, dim_size=dim_size, reduce="mul")
			Y = torch.pow(prod.cfloat(), 1/N)
			return Y.type(dtype)

	# number of neighbours
	N = scatter(torch.ones(x.shape[agg_dim], device=x.device), index, dim=agg_dim, dim_size=dim_size, reduce="sum")
	# Can't have divide by zero error, this should not affect final results
	N[N==0] = 1

	# when shift is True, p=0 is min, p=1 is avg, p=inf is max
	if shift:
		if torch.is_complex(x):
			x = x.real
		# TODO: handle empty x
		shifts, _ = x.min(dim=agg_dim)
		shifts -= 1e-3
		x = x - shifts
		Y = torch.exp(1 / p * (-torch.log(N)[:,None] + logsumexp_sparse(p * torch.log(x), index=index, agg_dim=agg_dim, dim_size=dim_size)))
		return (N[:,None] ** a) * (Y + shifts)
	# when shift is False, p=-inf is min, p=0 is prod, p=1 is avg, p=inf is max
	else:
		res = torch.exp((a * torch.log(N)[:,None]) + 1 / p * (-torch.log(N)[:,None] + logsumexp_sparse(p * torch.log(torch.abs(x)), index=index, agg_dim=agg_dim, dim_size=dim_size)))
		return res



def logsumexp_sparse(x, index, agg_dim=0, dim_size=None):
	"""log-sum-exp that works for both real and complex
	torch tensors"""
	dtype = x.dtype
	if torch.is_complex(x):
		c_max, _ = x.real.max(dim=agg_dim)
		c_min, _ = x.real.min(dim=agg_dim)
		x = x.cdouble()
	else:
		c_max, _ = x.max(dim=agg_dim)
		c_min, _ = x.min(dim=agg_dim)
		x = x.double()
	c = c_max
	c.unsqueeze(agg_dim)
	shifted = torch.exp(x - c)
	summed = scatter(shifted, index, dim=agg_dim, dim_size=dim_size, reduce="sum")
	res = c + torch.log(summed).type(dtype)
	return res




if __name__ == '__main__':
	in_size = 5
	# Shape [nodes, node_feats]
	x = torch.rand(in_size,2).cfloat()
	p = 0.00000001
	a = 1
	#res = gen_agg_sparse(x=x, p=p, index=index, shift=False)
	g = GenAgg(p=p, a=a, shift=False)
	res = g(x)

	# Shifted version
	g = GenAgg(p=p, shift=True)
	shifted = g(x)


	# Verify
	p = 0.00001
	# true_sum = (1 / x.shape[0] * ((x.cfloat() ** p).sum(dim=0))) ** (1 / p)
	prod = x.prod(dim=0)
	true_sum = torch.pow(prod, 1/x.shape[0])

	print("x:", x)
	print("genagg:", res)
	print("genagg (shifted):", shifted)
	print("prod:", prod)
	print("true sum:", true_sum)
