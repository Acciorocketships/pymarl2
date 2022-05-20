from pymarl.callbacks.callback import Callback
from pymarl.callbacks.local_reward_callback import LocalRewardCallback

REGISTRY = {}

REGISTRY["default_callback"] = Callback
REGISTRY["local_reward_callback"] = LocalRewardCallback