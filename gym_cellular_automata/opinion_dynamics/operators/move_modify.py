from collections.abc import Hashable

import numpy as np
from gym import logger, spaces

from gym_cellular_automata import Operator


def hashable(x):
    return isinstance(x, Hashable)


class Move(Operator):

    grid_dependant = True  # A minor effect because of boundaries
    action_dependant = True
    context_dependant = True

    deterministic = True

    def __init__(self, n_row, n_col, *args, **kwargs):

        super().__init__(*args, **kwargs)
        # fmt: off
        self.n_row = n_row
        self.n_col = n_col
        #self.not_move_set = None

        # fmt: on

        '''self.movement_set = (
            self.row #up_set
            | self.col#self.down_set
            #| #self.left_set
            #| #self.right_set
            | self.not_move_set
        )'''

    def update(self, grid, action, context):

        if not hashable(action):
            casting = int
            logger.warn(f"Unhashable Movement Action {action}.\nCasting to {casting}.")
            action = casting(action)
        if action is not None:
            row, col = action

            if (row < 0 or row > self.n_row) or (col < 0 or col  > self.n_col):

                logger.warn(
                    f"Movement Action {action} not in valid position")
        
        return grid, context
        



class Modify(Operator):
    hit = False

    grid_dependant = True
    action_dependant = True
    context_dependant = True

    deterministic = True

    def __init__(self, effects: dict, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.effects = effects

    def update(self, grid, action):
        self.hit = False     

        if action is not None:
            row, col = action
            
            grid[row,col] = self.effects
        return grid


class MoveModify(Operator):

    grid_dependant = True
    action_dependant = True
    context_dependant = True

    deterministic = True

    def __init__(self, move, modify, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.suboperators = move, modify

        self.move = move
        self.modify = modify

        if self.context_space is None:
            if (
                self.move.context_space is not None
                and self.modify.context_space is not None
            ):
                assert self.move.context_space == self.modify.context_space
                self.context_space = self.move.context_space

    def update(self, grid, action):
        #move_action, modify_action = subactions

        #grid, position = self.move(grid, move_action, position)
        grid = self.modify(grid, action)

        return grid, action
