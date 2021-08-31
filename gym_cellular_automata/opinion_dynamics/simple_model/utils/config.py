from gym_cellular_automata import PROJECT_PATH
from gym_cellular_automata.opinion_dynamics.utils.config import (
    get_config_dict,
    group_actions,
    translate,
)

CONFIG_PATH = (
    PROJECT_PATH / "./gym_cellular_automata/opinion_dynamics/simple_model/influencer_v0.yaml"
)


def get_GoL_config_dict():
    config = get_config_dict(CONFIG_PATH)

    return config


CONFIG = get_GoL_config_dict()
