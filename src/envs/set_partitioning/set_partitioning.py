from torch.distributions.gamma import Gamma
import torch
import numpy as np

from pymarl.envs.multiagentenv import MultiAgentEnv


class SetPartitioning(MultiAgentEnv):

	def __init__(self, batch_size=None, n_agents=16, n_actions=4, **kwargs):

		# User Params
		self.n_agents = n_agents
		self.n_actions = n_actions
		self.coalition_func = lambda x: np.mean(x, axis=-1)
		self.mix_func = lambda x: np.sum(x, axis=-1)
		self.reward_lambda = 1.
		self.empty_reward = -10.
		self.episode_limit = 1
		self.reset_characteristic = True
		self.reset_deterministic = True

		self.batch_mode = (batch_size != None)
		self.batch_size = batch_size if self.batch_mode else 1

		self.set_params(kwargs)
		self.coalition_func_full = lambda x: self.coalition_func(x) if len(x)>0 else self.empty_reward
		self.mix_func_full = lambda x: self.mix_func(x) if len(x)>0 else self.empty_reward

		# Internal Params

		self.reset_counter = 0

		self.reward_mat_dist = Gamma(2.0*torch.ones((self.batch_size,self.n_agents,self.n_actions)),
									1/self.reward_lambda*torch.ones((self.batch_size,self.n_agents,self.n_actions)))


		self.reward_mat = None
		self.obs = None
		self.t = 0
		self.adj = np.ones((self.batch_size, self.n_agents, self.n_agents))
		self.avail_actions = np.ones((self.batch_size, self.n_agents, self.n_actions))
		self.reset()


	def set_params(self, params):
		for name, value in params.items():
			setattr(self, name, value)


	def rectify_action(self, action):
		# action: batch x n_agents OR n_agents
		if not self.batch_mode:
			action = action[None,:]
		assert action.shape[0] == self.batch_size and action.shape[1] == self.n_agents, "incorrect action dimensions"
		# action: batch x n_agents x 1
		return np.array(action)[:,:,None]


	def step(self, action):
		# action: batch x n_agents x 1
		self.t += 1
		action = self.rectify_action(action)
		global_reward = self.get_reward(action)
		local_adv, all_rewards = self.local_rewards(action)
		terminated = np.array([(self.t >= self.episode_limit) for _ in range(self.batch_size)])
		if not self.batch_mode:
			global_reward = global_reward[0]
			terminated = terminated[0]
		return global_reward, terminated, {"local_rewards": local_adv}


	def get_reward(self, action):
		loc_rew = np.array([[self.coalition_func_full(
								np.take_along_axis(self.reward_mat[b,:,:], action[b,:,:], axis=-1)[action[b,:,:]==a])
							for a in range(self.n_actions)] for b in range(self.batch_size)])
		rew = self.coalition_func_full(loc_rew)
		return rew


	def local_rewards(self, action):
		def set_element(arr, idx, val):
			arr[:,idx] = val
			return arr
		orig_reward = self.get_reward(action)
		all_rewards = np.zeros((self.batch_size, self.n_agents, self.n_actions))
		max_other_rewards = np.zeros((self.batch_size, self.n_agents))
		for i in range(self.n_agents):
			# old_actioni = action[:,i]
			all_rewards[:,i,:] = np.array([self.get_reward(set_element(action, i, j)) for j in range(self.n_actions)]).T
			# max_other_rewards[:,i] = max(np.delete(all_rewards[:,i,:], old_actioni))
			max_other_rewards[:, i] = max(all_rewards[:, i, :])
			# action[:,i] = old_actioni
		return orig_reward - max_other_rewards, all_rewards

	def reset(self):
		if self.reset_characteristic or self.reward_mat is None:
			if self.reset_characteristic:
				rng_state = torch.get_rng_state()
				torch.random.manual_seed(self.reset_counter)
				self.reward_mat = self.reward_mat_dist.sample().numpy()
				torch.set_rng_state(rng_state)
				self.reset_counter += 1
			else:
				self.reward_mat = self.reward_mat_dist.sample().numpy()
		self.t = 0
		return self.get_obs()

	def get_obs(self, batch=0):
		if not self.batch_mode:
			return self.get_obs_agent(agent_id=slice(None), batch=batch)
		else:
			return self.get_obs_agent(agent_id=slice(None))

	def get_obs_agent(self, agent_id=slice(None), batch=slice(None)):
		return self.reward_mat[batch, agent_id, :]

	def get_obs_size(self):
		return self.n_actions

	def get_info(self, batch=0):
		return {"adj": self.adj[batch]}

	def get_state(self, batch=slice(None)):
		return self.reward_mat[batch].reshape(-1, self.get_state_size())

	def get_state_size(self):
		return self.n_agents * self.n_actions

	def get_avail_agent_actions(self, agent_id=slice(None), batch=slice(None)):
		return self.avail_actions[batch, agent_id, :]

	def get_avail_actions(self, batch=0):
		if not self.batch_mode:
			return self.get_avail_agent_actions(agent_id=slice(None), batch=batch)
		else:
			return self.get_avail_agent_actions(agent_id=slice(None))

	def get_total_actions(self):
		return self.n_actions

	def get_stats(self):
		return {}

	def close(self):
		pass

	def render(self):
		pass

	def seed(self):
		pass

	def save_replay(self):
		pass

	def metrics(self, env_data, model_data, **kwargs):
		metrics = {}

		mask = env_data["filled"][:, :-1].float()
		local_rewards = env_data['local_rewards'][:, :-1]
		local_values = model_data["local_q_chosen"]
		B, T, n_agents = local_rewards.shape
		local_rewards = local_rewards.view(B * T, n_agents)
		local_values = local_values.view(B * T, n_agents)
		sbs = torch.stack([local_rewards, local_values], dim=-1)
		corrs = np.array([np.corrcoef(sbs[i, :, :], rowvar=False)[0, 1] for i in range(B * T)])
		corrs = corrs * mask.view(B * T)
		corr = np.sum(corrs) / np.sum(mask)
		metrics["local_reward_corr"] = corr

		return metrics


if __name__ == '__main__':
	env = SetPartitioning()
	act = np.random.randint(low=0, high=env.n_actions, size=env.n_agents)
	reward, done, info = env.step(act)
	# print(env.reward_mat)
	# env.reset()
	# print(env.reward_mat)
	import pdb; pdb.set_trace()