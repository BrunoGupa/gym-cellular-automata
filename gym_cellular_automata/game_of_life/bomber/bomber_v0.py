import numpy as np
from gym import logger, spaces
import matplotlib.pyplot as plt
from gym_cellular_automata import CAEnv, GridSpace, Operator
from gym_cellular_automata.game_of_life.operators import (
    GoL,
    Modify,
    Move,
    MoveModify,
)

from .utils.config import CONFIG
from .utils.render import add_helicopter, plot_grid


class GoLEnvBomberV0(CAEnv):
    metadata = {"render.modes": ["human"]}

    # fmt: off
    _dead             = CONFIG["cell_symbols"]["dead"]
    _alive            = CONFIG["cell_symbols"]["alive"]


    _row              = CONFIG["grid_shape"]["n_row"]
    _col              = CONFIG["grid_shape"]["n_col"]

    _p_live           = CONFIG["ca_params"]["p_live"]
    _effects          = CONFIG["effects"]

    _max_freeze       = CONFIG["max_freeze"]
    _n_actions        = CONFIG["grid_shape"]["n_row"] * CONFIG["grid_shape"]["n_col"]
    _max_steps        = CONFIG["episode_params"]["max_steps"]        

    _reward_per_dead  = CONFIG["rewards"]["per_dead"]
    _reward_per_alive = CONFIG["rewards"]["per_alive"]
    # fmt: on

    def __init__(self, rows=None, cols=None, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self._row = self._row if rows is None else rows
        self._col = self._col if cols is None else cols



        self._set_spaces()

        self.cellular_automaton = GoL(
            self._dead, self._alive, **self.ca_space
        )
        self.move = Move(self._row, self._col, **self.move_space)
        self.modify = Modify(self._effects, **self.modify_space)
    
        self.move_modify = MoveModify(self.move, self.modify, **self.move_modify_space)

        # Composite Operators
        self._MDP = MDP(
            self.cellular_automaton,
            self.move_modify,
            self._max_freeze,
            **self.MDP_space,
        )

    @property
    def MDP(self):
        return self._MDP

    @property
    def initial_state(self):

        if self._resample_initial:

            self.grid = self.grid_space.sample()

            self.old_grid = None

            ca_params = np.array([self._p_live])
            pos = np.array([self._row // 2, self._col // 2])
            freeze = np.array(self._max_freeze)
            self.context = ca_params, pos, freeze

            self._initial_state = self.grid, self.context

        self._resample_initial = False

        return self._initial_state

    def _award(self):
        dict_counts = self.count_cells(self.grid)
        
        cell_counts = np.array(
            [dict_counts[self._dead], dict_counts[self._alive]]
        )

        reward_weights = np.array(
            [self._reward_per_dead, self._reward_per_alive]
        )

        return np.dot(reward_weights, cell_counts)

    def _is_done(self):
        self.done = not bool(np.any(self.grid == self._alive))
        if self.steps_beyond_done == 0:
            self.done = True 


    def _report(self):
        return {"hit": self.modify.hit}

    def render(self, mode="human"):

        if mode == "human":

            _, pos, _ = self.context


            if pos is None:
                return plot_grid(self.grid)
            else:
                if self.old_grid is None:
                    add_helicopter(plot_grid(self.grid, True), pos)
                    return plt.gcf()
                else:
                    add_helicopter(plot_grid(self.old_grid, True), pos), add_helicopter(plot_grid(self.grid, True), pos)
                    return plt.gcf()
        else:

            logger.warn(
                f"Undefined mode.\nAvailable modes {self.metadata['render.modes']}"
            )

    def _set_spaces(self):
        self.ca_params_space = spaces.Box(np.array([0,0]), np.array([self._row, self._col]))
        self.position_space = spaces.MultiDiscrete([self._row, self._col])
        self.freeze_space = spaces.Discrete(self._max_freeze + 1)

        self.context_space = spaces.Tuple(
            (self.ca_params_space, self.position_space, self.freeze_space)
        )

        self.grid_space = GridSpace(
            values=[self._dead, self._alive],
            shape=(self._row, self._col),
        )

        # RL spaces

        self.action_space = spaces.Tuple((spaces.Discrete(self._row), spaces.Discrete(self._col)))
        self.observation_space = spaces.Tuple((self.grid_space, self.context_space))

        # Suboperators Spaces

        self.ca_space = {
            "grid_space": self.grid_space,
            "action_space": self.action_space,
            "context_space": self.ca_params_space,
        }

        self.move_space = {
            "grid_space": self.grid_space,
            "action_space": self.action_space,
            "context_space": self.position_space,
        }

        self.modify_space = {
            "grid_space": self.grid_space,
            "action_space": spaces.Discrete(2),
            "context_space": self.position_space,
        }

        self.move_modify_space = {
            "grid_space": self.grid_space,
            "action_space": spaces.Tuple((self.action_space, spaces.Discrete(2))),
            "context_space": self.position_space,
        }

        self.MDP_space = {
            "grid_space": self.grid_space,
            "action_space": self.action_space,
            "context_space": self.context_space,
        }


class MDP(Operator):
    from collections import namedtuple

    Suboperators = namedtuple("Suboperators", ["cellular_automaton", "move_modify"])

    grid_dependant = True
    action_dependant = True
    context_dependant = True

    deterministic = False

    def __init__(self, cellular_automaton, move_modify, max_freeze, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.move_modify = move_modify
        self.ca = cellular_automaton

        self.suboperators = self.Suboperators(cellular_automaton, move_modify)

        self.max_freeze = max_freeze
        self.freeze_space = spaces.Discrete(max_freeze + 1)

    def update(self, grid, action, context):

        ca_params, _, freeze = context


        if freeze == 0:

            grid, position = self.move_modify(grid, action)
            old_grid = grid.copy()
            grid, ca_params = self.ca(grid, None, ca_params)
            
            freeze = np.array(self.max_freeze)

        else:

            grid, position = self.move_modify(grid, action)
            old_grid = None

            freeze = np.array(freeze - 1)

        context = ca_params, position, freeze

        return old_grid, grid, context
