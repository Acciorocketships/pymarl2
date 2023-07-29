from functools import partial
import sys
import os

from .multiagentenv import MultiAgentEnv
from .stag_hunt import StagHunt
from .matrix_game import OneStepMatrixGame
from .estimate_game import EstimateGame
from .set_partitioning import SetPartitioning

try:
    vmasenv = True
    from pymarl.envs.vmas_wrapper import get_vmas_env
except Exception as e:
    print(e)
    vmasenv = False


try:
    starcraftenv = True
    from .starcraft import StarCraft2Env
except Exception as e:
    print(e)
    starcraftenv = False

try:
    gfootball = True
    from .gfootball import GoogleFootballEnv
except:
    gfootball = False

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["set"] = partial(env_fn, env=SetPartitioning)
REGISTRY["estimate"] = partial(env_fn, env=EstimateGame)
REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)
REGISTRY["one_step_matrix_game"] = partial(env_fn, env=OneStepMatrixGame)

if vmasenv:
    REGISTRY["vmas"] = get_vmas_env

if starcraftenv:
    REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)

if gfootball:
    REGISTRY["gfootball"] = partial(env_fn, env=GoogleFootballEnv)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", "~/StarCraftII")
