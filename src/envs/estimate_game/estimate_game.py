from envs.multiagentenv import MultiAgentEnv
import numpy as np

class EstimateGame(MultiAgentEnv):

	def __init__(self, batch_size=None, n_agents=8, **kwargs):
		# Params
		self.batch_mode = (batch_size != None)
		self.n_agents = n_agents
		self.batch_size = batch_size if self.batch_mode else 1
		self.mixfunc = lambda x: np.min(x, axis=-1)
		self.n_actions = 10
		self.episode_limit = 1
		self.density = 0.2
		self.localperc = 0.3

		self.state = None
		self.adjacency = None
		self.t = 0


	def local_reward(self, state, action):
		# state: batch x n_agents x 1
		# action: batch x n_agents x 1
		gap = 1 / self.n_actions
		guess = action * gap + gap/2
		error = np.maximum(np.abs(guess - state) - gap/2, 0)
		# error: batch x n_agents x 1
		return -error


	def mix_states(self, state, adjacency):
		# state: batch x n_agents x 1
		# adjacency: batch x n_agents x n_agents

		diag_indices = np.arange(self.n_agents)
		adjacency[:,diag_indices,diag_indices] = 0
		neighbour_states = adjacency[:,:,:,None] * state[:,None,:,:]
		degree = np.sum(adjacency, axis=-2)[:,:,None]
		degree[degree==0] = 1
		neighbour_states_sum = np.sum(neighbour_states, axis=-2) / degree
		mixed_state = self.localperc * state + (1 - self.localperc) * neighbour_states_sum
		new_state = 1.4 / (self.localperc + 0.4) * (mixed_state - 0.5) + 0.5
		new_state = np.minimum(new_state, 1.0)
		new_state = np.maximum(new_state, 0.0)
		# new_state: batch x n_agents x state_dim
		return new_state


	def gen_adjacency(self):
		orig_density = np.sqrt(self.density)
		A = np.random.rand(self.batch_size, self.n_agents, self.n_agents) < orig_density
		A = A * A.transpose((0,2,1))
		diag_indices = np.arange(self.n_agents)
		A[:,diag_indices,diag_indices] = 1
		# A: batch x n_agents x n_agents
		return A.astype(int)


	def gen_state(self):
		# state: batch x n_agents x 1
		return np.random.rand(self.batch_size, self.n_agents, 1)


	def rectify_actions(self, actions):
		# actions: n_agents x batch
		if not self.batch_mode:
			actions = actions[:,None]
		assert actions.shape[0] == self.n_agents and actions.shape[1] == self.batch_size, "incorrect actions dimensions"
		actions = actions.transpose((1,0))[:,:,None]
		# actions: batch x n_agents x 1
		return actions


	def step(self, actions):
		self.t += 1
		actions = self.rectify_actions(actions)
		mixed_state = self.mix_states(self.state, self.adjacency)
		qind = self.local_reward(mixed_state, actions)[:,:,0]
		qglobal = self.mixfunc(qind)
		terminated = [(self.t >= self.episode_limit) for _ in range(self.batch_size)]
		if not self.batch_mode:
			qglobal = qglobal[0].item()
			terminated = terminated[0]
		return qglobal, terminated, {"adj": self.adjacency}


	def reset(self):
		self.state = self.gen_state()
		self.adjacency = self.gen_adjacency()
		self.t = 0
		return self.get_obs()


	def get_obs_agent(self, agent_id, batch=0):
		return self.state[batch, agent_id]


	def get_obs(self):
		return [self.get_obs_agent(i) for i in range(self.n_agents)]


	def get_obs_size(self):
		return 1


	def get_state(self):
		return self.state[0].reshape(self.get_state_size())


	def get_state_size(self):
		return self.n_agents


	def get_avail_agent_actions(self, agent_id):
		return np.ones(self.n_actions)


	def get_avail_actions(self):
		return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]


	def get_total_actions(self):
		return self.n_actions


	def get_stats(self):
		return None


	def close(self):
		pass



# from envs import EstimateGame
# import numpy as np
# m = EstimateGame()
# x = m.gen_state()
# a = m.gen_adjacency()

# xp = m.mix_states(x, a)

# act = np.random.randint(low=0, high=m.n_actions, size=(m.batch_size, m.n_agents, 1))
# rew = m.local_reward(xp, act)
	