import numpy as np
from matplotlib import pyplot as plt
from collections import Counter

import gym
from gym import spaces
from gym.utils import seeding

from .operators import (
    ForestFireCellularAutomaton,
    ForestFireModifier,
    ForestFireCoordinator,
)
from .utils.config import get_forest_fire_config_dict
from .utils.render import plot_grid, add_helicopter
from .utils.initializers import init_bulldozer, generate_wind_kernel

CONFIG = get_forest_fire_config_dict()

CELL_STATES = CONFIG["cell_states"]

ROW = CONFIG["grid_shape"]["n_row"]
COL = CONFIG["grid_shape"]["n_col"]

FIRES = CONFIG["ca_params"]["fires"]
INIT_P_TREE = CONFIG["ca_params"]["init_p_tree"]
WIND_SPEED = CONFIG["ca_params"]["wind_speed"]
WIND_DIRECTION = CONFIG["ca_params"]["wind_direction"]
WIND_C1 = CONFIG["ca_params"]["wind_c1"]

EFFECTS = CONFIG["effects"]

TIMES = CONFIG["mdp_internal_times"]
MAX_TIME = max(TIMES.values())

ACTION_MIN = CONFIG["actions"]["min"]
ACTION_MAX = CONFIG["actions"]["max"]

# spaces.Box requires typing for discrete values
CELL_TYPE = CONFIG["cell_type"]
ACTION_TYPE = CONFIG["action_type"]

# ------------ Forest Fire Environment

# NEW CONTEXT
# bulldozer's position, internal mdp's time, alive status

class BulldozerEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    empty = CONFIG["cell_symbols"]["empty"]
    tree = CONFIG["cell_symbols"]["tree"]
    fire = CONFIG["cell_symbols"]["fire"]
    burnt = CONFIG["cell_symbols"]["burnt"]

    pos_space = spaces.MultiDiscrete([ROW, COL])
    times_space = spaces.Discrete(MAX_TIME + 1)
    alive_space = spaces.Discrete(2)

    context_space = spaces.Tuple((pos_space, times_space, alive_space))
    grid_space = spaces.Box(0, CELL_STATES - 1, shape=(ROW, COL), dtype=CELL_TYPE)

    action_space = spaces.Box(ACTION_MIN, ACTION_MAX, shape=tuple(), dtype=ACTION_TYPE)
    observation_space = spaces.Tuple((grid_space, context_space))

    wind = generate_wind_kernel(WIND_DIRECTION, WIND_SPEED, WIND_C1)
    last_counts = None

    def __init__(self):

        self.cellular_automaton = ForestFireCellularAutomaton(
            grid_space=self.grid_space,
            wind=self.wind,
            action_space=self.action_space,
        )

        self.modifier = ForestFireModifier(
            EFFECTS,
            TIMES,
            endCells = {self.fire},
            grid_space=self.grid_space,
            action_space=self.action_space,
            context_space=self.context_space,
        )

        self.coordinator = ForestFireCoordinator(
            self.cellular_automaton, self.modifier, mdp_time = TIMES,
            context_space = self.context_space
        )

        self.reward_per_empty = CONFIG["rewards"]["per_empty"]
        self.reward_per_tree = CONFIG["rewards"]["per_tree"]
        self.reward_per_fire = CONFIG["rewards"]["per_fire"]
        self.reward_per_burnt = CONFIG["rewards"]["per_burnt"]
        self.reward_cut = CONFIG["rewards"]["cut"]
        self.reward_contact = CONFIG["rewards"]["fire_contact"]
        print("New bulldozer env created, kernel\n {}".format(self.wind))

    def reset(self):
        self.grid, pos = init_bulldozer(self.grid_space,
                                        self.empty,
                                        self.tree,
                                        self.fire,
                                        INIT_P_TREE,
                                        fires = FIRES)
        # NEW CONTEXT
        # bulldozer's position, internal mdp's time, alive status
        self.context = pos, 0, True
        self.coordinator.last_lattice_update = 0
        self.modifier.contact = False
        self.last_counts = None

        obs = self.grid, self.context

        return obs

    def step(self, action):
        done = self._is_done()

        if not done:

            new_grid, new_context = self.coordinator(self.grid, action, self.context)

            self.grid = new_grid
            self.context = new_context

        obs = self.grid, self.context
        reward = self._award() if not done else 0.0
        info = self._report()

        return obs, reward, done, info

    def _award(self):
        dict_counts = Counter(self.grid.flatten().tolist())

        new_burnt = dict_counts[self.burnt] - self.last_counts[self.burnt]
        self.last_counts = dict_counts

        reward = new_burnt * self.reward_per_burnt
        reward += self.modifier.contact * self.reward_contact
        reward += self.modifier.hit * self.reward_cut

        return reward

    def _is_done(self):
        _, _, alive = self.context
        if self.last_counts is None:
            self.last_counts = Counter(self.grid.flatten().tolist())
        if self.last_counts[self.fire] == 0 or not alive:
            return True
        return False

    def _report(self):
        return {"hit": self.modifier.hit, "alive": not self.modifier.contact}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode="human"):
        pos, _, _ = self.context

        figure = add_helicopter(plot_grid(self.grid), pos)
        plt.show()

        return figure
