from pymarl.modules.agents import REGISTRY as agent_REGISTRY
from pymarl.components.action_selectors import REGISTRY as action_REGISTRY
from pymarl.controllers.basic_controller import BasicMAC
from pymarl.utils.rl_utils import RunningMeanStd
import torch as th
import numpy as np
from torch_geometric.data import Batch

# This multi-agent controller shares parameters between agents and includes the info dict from the environment
class InfoMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        self.existing_keys = set(['state', 'obs', 'actions', 'avail_actions', 'probs', 'reward', 'terminated', 'actions_onehot', 'filled'])
        
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        qvals = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(qvals[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def build_info(self, batch, t):
        info = {key: val[:,t] for key, val in batch.data.transition_data.items() if key not in self.existing_keys}
        for key, val in info.items():
            if isinstance(val, np.ndarray) and val.dtype=='object':
                info[key] = Batch.from_data_list(val)
        return info

    def forward(self, ep_batch, t, test_mode=False):
        if test_mode:
            self.agent.eval()

        info = self.build_info(ep_batch, t)
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states, info=info)

        return agent_outs