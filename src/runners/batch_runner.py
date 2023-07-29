import numpy as np
import torch
from functools import partial
from torch_geometric.data import Data
from pymarl.envs import REGISTRY as env_REGISTRY
from pymarl.components.episode_buffer import EpisodeBatch



class BatchRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0
        self.t_env = 0

        self.env_info = self.env.get_env_info()

        self.info_scheme = {}
        self.set_info_scheme()

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = np.zeros(self.batch_size, dtype=bool)
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated.all():

            mask = np.logical_not(terminated)

            pre_info = {key: val[mask] for key, val in self.env.get_info(batch=slice(None)).items()}

            pre_transition_data = {
                "state": self.env.get_state(batch=slice(None))[mask],
                "avail_actions": self.env.get_avail_actions(batch=slice(None))[mask],
                "obs": self.env.get_obs(batch=slice(None))[mask],
                **pre_info,
            }

            self.batch.update(pre_transition_data, bs=mask, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            # Fix memory leak
            cpu_actions = actions.to("cpu").numpy()

            self.env.test_mode = test_mode
            reward, terminated, post_info = self.env.step(actions)
            episode_return += reward

            post_transition_data = {
                "actions": cpu_actions[mask],
                "reward": reward[mask],
                "terminated": terminated[mask],
                **{key: val[mask] for key, val in post_info.items()}
            }

            self.batch.update(post_transition_data, bs=mask, ts=self.t)

            self.t += 1

        last_data = {
            "state": self.env.get_state(batch=slice(None)),
            "avail_actions": self.env.get_avail_actions(batch=slice(None)),
            "obs": self.env.get_obs(batch=slice(None)),
            **self.env.get_info(batch=slice(None)),
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        # Fix memory leak
        cpu_actions = actions.to("cpu").numpy()
        self.batch.update({"actions": cpu_actions}, ts=self.t)
        
        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + post_info.get(k, 0) for k in set(cur_stats) | set(post_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode:
            self._log(cur_returns, cur_stats, log_prefix)
            self.log_replay()
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()

    def log_replay(self):
        if self.args.save_replay:
            frame_list = self.env.save_replay()
            self.logger.log_video(frame_list, t=self.t_env)


    def set_info_scheme(self):
        # self.env.reset()
        pre_info = self.env.get_info()
        def add_dict(d):
            for key, val in d.items():
                item_scheme = {}
                if isinstance(val, Data):
                    x_shape = val.x.shape[1:] if (val.x is not None) else 0
                    edge_shape = val.edge_attr.shape[1:] if (val.edge_attr is not None) else 0
                    item_scheme['vshape'] = (x_shape, edge_shape)
                    item_scheme['dtype'] = Data
                else:
                    val = np.array(val)
                    item_scheme['vshape'] = val.shape[1:]
                    if len(val.shape) > 0:
                        if val.shape[0] == self.env_info['n_agents']:
                            item_scheme['group'] = 'agents'
                    item_scheme['dtype'] = torch.tensor(np.empty(0, dtype=val.dtype)).dtype
                self.info_scheme[key] = item_scheme
        add_dict(pre_info)
        avail_actions = self.env.get_avail_actions()
        actions = np.argmax(avail_actions, axis=-1)
        reward, done, post_info = self.env.step(actions)
        add_dict(post_info)
