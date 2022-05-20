from pymarl.callbacks.callback import Callback
import numpy as np
import torch

class LocalRewardCallback(Callback):

    def __init__(self):
        super().__init__()

    def metrics(self, env_data, model_data, **kwargs):
        metrics = {}
        local_rewards = env_data['local_rewards'][:, :-1].cpu()
        global_rewards = env_data['reward'][:, :-1].cpu()
        local_values = model_data["local_q_chosen"].cpu()
        B, T, n_agents = local_rewards.shape
        local_rewards = local_rewards.view(B * T, n_agents).numpy()
        local_values = local_values.view(B * T, n_agents).numpy()
        local_sbs = np.stack([local_rewards, local_values], axis=-1)
        corrs = np.array([np.corrcoef(local_sbs[i, :, :], rowvar=False)[0, 1] for i in range(B * T)])
        corrs = corrs[~np.isnan(corrs)]
        local_corr = np.mean(corrs)
        metrics["localq_localr_corr"] = local_corr
        local_rewards_var = np.mean(np.var(local_rewards, axis=-1))
        local_values_var = np.mean(np.var(local_values, axis=-1))
        metrics["local_reward_var"] = local_rewards_var
        metrics["local_q_var"] = local_values_var
        global_rewards_flat = global_rewards.repeat((1,1,n_agents)).reshape(B*T*n_agents)
        local_values_flat = local_values.reshape(B*T*n_agents)
        global_sbs = np.stack([global_rewards_flat, local_values_flat], axis=-1)
        global_corr = np.corrcoef(global_sbs, rowvar=False)[0,1]
        metrics["localq_globalr_corr"] = global_corr
        return metrics