import numpy as np
from gym import spaces

from gym_cellular_automata import Operator
from gym_cellular_automata.game_of_life.utils.neighbors import neighborhood_at


class GoL(Operator):

    grid_dependant = True
    action_dependant = True
    context_dependant = True

    deterministic = False

    def __init__(self, dead, alive, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.dead = dead
        self.alive = alive

        if self.context_space is None:
            self.context_space = spaces.Box(0.0, 1.0, shape=(2,))

    def update(self, grid, action, context):
        # A copy is needed for the sequential update of a CA
        new_grid = grid.copy()
        #new_grid = flip(action, new_grid)

        for row, cells in enumerate(grid):
            for col, cell in enumerate(cells):

                neighbors = neighborhood_at(grid, (row, col), invariant=self.dead)

                num_neighbors = np.sum(neighbors) - cell

                if cell and not 2<= num_neighbors <= 3:
                    new_grid[row][col] = 0
                
                elif num_neighbors == 3:
                    new_grid[row][col] = 1

        return new_grid, context
