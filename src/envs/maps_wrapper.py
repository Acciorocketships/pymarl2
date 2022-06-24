import torch
import numpy as np
from PIL import Image
from maps import make_env

class MapsWrapper(object):

    def __init__(self, env, render=False, save_replay=False):
        self.env = env
        self._obs = None
        self._rew = None
        self._done = None
        self._info = None
        self.test_mode = False
        self.show_frames = render
        self.save_frames = save_replay
        self.frame_list = []
        self.batch_size = self.env.num_envs
        self.obs_size = self.env.observation_space[0].shape[0]
        self.n_agents = self.env.n_agents
        self.episode_limit = self.env.max_steps
        if self.env.continuous_actions:
            self.action_size = self.env.action_space[0].shape[0]
        else:
            self.action_size = self.env.action_space[0].n
        self.reset()

    def step(self, actions):
        actions = actions.transpose(1,0) # batch x n_agents x action_dim to n_agents x batch x action_dim
        if len(actions.shape) == 2:
            actions = actions.unsqueeze(2)
        obs, rew, done, info = self.env.step(actions)
        self.render()
        local_rew = self.vectorise_obs(rew)
        self._rew = np.mean(local_rew, axis=-1)
        self._obs = self.vectorise_obs(obs)
        self._done = done.cpu().numpy()
        self._info = self.vectorise_infos(info)
        return self._rew, self._done, self._info

    def reset(self):
        obs = self.env.reset()
        self._obs = self.vectorise_obs(obs)
        info_list = [self.env.scenario.info(agent) for agent in self.env.agents]
        self._info = self.vectorise_infos(info_list)
        self.frame_list = []
        return self._obs

    def get_obs(self, batch=slice(None)):
        return self._obs[batch]

    def get_obs_agent(self, agent_id, batch=slice(None)):
        return self._obs[batch,agent_id,:]

    def get_obs_size(self):
        return self.obs_size

    def get_state(self, batch=slice(None)):
        return self._obs.reshape(self.batch_size, self.n_agents * self.obs_size)[batch]

    def get_info(self, batch=slice(None)):
        if batch == slice(None):
            return self._info
        else:
            return {key: val[batch] for key, val in self._info.items()}

    def get_state_size(self):
        return self.n_agents * self.get_obs_size()

    def get_avail_actions(self, batch=slice(None)):
        return torch.ones(self.batch_size, self.n_agents, self.action_size)

    def get_avail_agent_actions(self, agent_id, batch=slice(None)):
        return torch.ones(self.batch_size, self.action_size)

    def get_total_actions(self):
        return self.action_size

    def render(self):
        if self.show_frames or (self.save_frames and self.test_mode):
            if (self.save_frames and self.test_mode):
                img = self.env.render(mode="rgb_array", visualize_when_rgb=self.show_frames,)
                self.frame_list.append(img)
            else:
                self.env.render(mode="human")

    def close(self):
        pass

    def seed(self):
        return self.env.seed()

    def save_replay(self):
        return np.array(self.frame_list)

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    def vectorise_infos(self, infos):
        return {key: torch.stack([infos[i][key].cpu().numpy() for i in range(self.n_agents)], dim=1) for key in infos[0].keys()}

    def vectorise_obs(self, obs):
        return torch.stack(obs, dim=1).cpu().numpy()



def get_maps_env(**kwargs):
    def delete(dictionary, key):
        if key in dictionary:
            del dictionary[key]
    map_name = kwargs.get("map_name", None)
    batch_size = kwargs.get("batch_size", 1)
    device = kwargs.get("device", "cpu")
    continuous = kwargs.get("continuous", False)
    save_replay = kwargs.get("save_replay", False)
    render = kwargs.get("render", False)
    assert map_name is not None, "must specify env_args.map_name"
    env_kwargs = kwargs.copy()
    delete(env_kwargs, "map_name")
    delete(env_kwargs, "batch_size")
    delete(env_kwargs, "device")
    delete(env_kwargs, "continuous")
    delete(env_kwargs, "save_replay")
    delete(env_kwargs, "render")
    env = make_env(
        map_name,
        num_envs=batch_size,
        device=device,
        continuous_actions=continuous,
        rllib_wrapped=False,
        # Environment specific variables
        **env_kwargs,
    )
    wrapped_env = MapsWrapper(env, render=render, save_replay=save_replay)
    return wrapped_env
