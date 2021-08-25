import warnings
from pathlib import Path

from gym.envs.registration import register
from gym.error import Error as GymError

from gym_cellular_automata.ca_env import CAEnv
from gym_cellular_automata.grid_space import GridSpace
from gym_cellular_automata.operator import Operator

# Ignore warnings trigger by Bulldozer Render
# EmojiFont raises RuntimeWarning
warnings.filterwarnings("ignore", message="Glyph 108 missing from current font.")
warnings.filterwarnings("ignore", message="Glyph 112 missing from current font.")

# Global path on current machine
PROJECT_PATH = Path(__file__).parents[1]

REGISTERED_CA_ENVS = (
    "ForestFireHelicopter-v0",
    "ForestFireBulldozer-v1",
    "GoLBomber-v0",
    "Influencer-v0"
)

try:
    ffdir = "gym_cellular_automata"
    register(
        id=REGISTERED_CA_ENVS[0],
        entry_point=ffdir + ".forest_fire.helicopter:ForestFireEnvHelicopterV0",
        max_episode_steps=999,
    )

    register(
        id=REGISTERED_CA_ENVS[1],
        entry_point=ffdir + ".forest_fire.bulldozer:ForestFireEnvBulldozerV1",
        max_episode_steps=4096,
        reward_threshold=0.0,
    )

    register(
        id=REGISTERED_CA_ENVS[2],
        entry_point=ffdir + ".game_of_life.bomber:GoLEnvBomberV0",
        max_episode_steps=100,
        #reward_threshold=1000,
    )

    register(
        id=REGISTERED_CA_ENVS[3],
        entry_point=ffdir + ".opinion_dynamics.simple_model:InfluencerEnvV0",
        max_episode_steps=100,
        #reward_threshold=1000,
    )
except GymError:  # Avoid annoying Re-register error when working interactively.
    pass


__all__ = ["CAEnv", "Operator", "GridSpace", "REGISTERED_CA_ENVS"]
