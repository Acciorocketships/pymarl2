import torch
import torch.nn as nn
import numpy as np
from functools import partial
from modules.layer.mixer import Mixer, AggMixer
from modules.layer.mlp import MLP, layers
from utils.param_update import update_module_params, get_num_params, batch_linear, LinearHyper, abs_linear


class QGNNMixer(nn.Module):
	def __init__(self, args):
		super().__init__()

		self.args = args
		self.n_agents = args.n_agents
		self.state_dim = int(np.prod(args.state_shape))
		self.embed_dim = args.mixing_embed_dim
		self.mixer_nlayers = args.mixer_nlayers
		self.mixer_midmult = args.mixer_midmult
		self.hypernet_flat = getattr(args, "hypernet_flat", True)
		self.use_hypernet = getattr(args, "use_hypernet", False)
		self.use_layernorm = getattr(args, "mixer_use_layernorm", False)
		self.use_genagg = getattr(args, "use_genagg", True)
		
		if self.use_genagg:
			self.mixer = AggMixer(input_dim=1, hidden_dim=self.embed_dim, output_dim=1, 
								  psi_layers=self.mixer_nlayers, phi_layers=self.mixer_nlayers, 
								  midmult=self.mixer_midmult, layernorm=self.use_layernorm)
		else:
			self.mixer = Mixer(input_dim=1, hidden_dim=self.embed_dim, output_dim=1, 
							   psi_layers=self.mixer_nlayers, phi_layers=self.mixer_nlayers, 
							   midmult=self.mixer_midmult, layernorm=self.use_layernorm)

		self.num_params_mixer = get_num_params(self.mixer)

		if self.use_hypernet:
			if self.hypernet_flat:
				self.hypernet = MLP(input_dim=self.state_dim, output_dim=self.num_params_mixer, 
									layer_sizes=layers(input_dim=self.state_dim, output_dim=self.num_params_mixer, nlayers=2, midmult=1.), 
									layernorm=self.use_layernorm)
			else:
				self.state_dim /= self.n_agents
				self.hypernet = AggMixer(input_dim=self.state_dim, hidden_dim=self.state_dim, 
										 output_dim=self.num_params_mixer, layernorm=self.use_layernorm)


	def forward(self, q_values, states):
		print("mixer")
		# q_values: batch x episode_len x n_agents
		# states: batch x episode_len x n_agents x obs_shape
		# output: batch x episode_len x 1
		B, episode_len, n_agents = q_values.shape
		if self.use_hypernet:
			if self.hypernet_flat:
				states = states.reshape(B, episode_len, self.state_dim)
			else:
				states = states.reshape(B, episode_len, self.n_agents, self.state_dim)
			params = self.hypernet(states)
			params = params.reshape(B * episode_len, self.num_params_mixer)
			update_module_params(module=self.mixer, params=params, param_dim=1,
								 filter_cond=lambda module: isinstance(module, nn.Linear) or isinstance(module, LinearHyper),
								 replace_func=partial(batch_linear, absweight=True))
		else:
			update_module_params(module=self.mixer, filter_cond=lambda module: isinstance(module, nn.Linear), replace_func=abs_linear)
		q_values = q_values.reshape(B * episode_len, self.n_agents, 1)
		global_q = self.mixer(q_values)
		global_q = global_q.reshape(B, episode_len, 1)
		return global_q