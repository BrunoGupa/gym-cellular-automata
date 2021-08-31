import numpy as np
from gym import spaces

from gym_cellular_automata import Operator
from gym_cellular_automata.opinion_dynamics.utils.neighbors import neighborhood_at, neighborhood_vonNeumann


class OpinionCA(Operator):

    grid_dependant = True
    action_dependant = True
    context_dependant = True

    deterministic = False

    def __init__(self, n, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.num_opinions = n

        if self.context_space is None:
            self.context_space = spaces.Box(0.0, 1.0, shape=(2,))

    def update(self, grid, action, context, vonNeumann=True):
        # A copy is needed for the sequential update of a CA
        new_grid = grid.copy()
        #new_grid = flip(action, new_grid)
        
        for row, cells in enumerate(grid):
            for col, cell in enumerate(cells):
                #use the vonNeumann neighborhood
                if vonNeumann:
                    neighbors, size_neighborhood = neighborhood_vonNeumann(grid, (row, col))
                else:
                    #uste the Moore neighborhood
                    neighbors, size_neighborhood = neighborhood_at(grid, (row, col))

                avarage_neighbors = np.sum(neighbors) / ((self.num_opinions-1) * size_neighborhood)

                new_grid[row][col] = self.step_func(avarage_neighbors, self.num_opinions)


                
                
        return new_grid, context
    
    def step_func(self,x ,n):
        num = np.floor(x*n)
        if num > n-1:
            return (n-1) * 1.0
        return num
