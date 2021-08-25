from gym_cellular_automata import PROJECT_PATH
from gym_cellular_automata.game_of_life.utils.config import (
    get_config_dict,
    group_actions,
    translate,
)

CONFIG_PATH = (
    PROJECT_PATH / "./gym_cellular_automata/game_of_life/bomber/bomber_v0.yaml"
)


def get_GoL_config_dict():
    config = get_config_dict(CONFIG_PATH)

    return config


CONFIG = get_GoL_config_dict()
