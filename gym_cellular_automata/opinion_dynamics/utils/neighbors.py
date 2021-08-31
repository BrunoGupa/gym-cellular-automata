from typing import Union

import numpy as np



def moore_n(
    n: int, position: tuple, grid: np.ndarray, invariant: Union[int, np.ndarray] = 0
):
    """Gets the N Moore neighborhood at given postion."""

    row, col = position
    nrows, ncols = grid.shape

    # Target offsets from position.
    ofup, ofdo = row + np.array([-n, +n])
    ofle, ofri = col + np.array([-n, +n])

    try:

        if ofup < 0 or ofle < 0 or ofdo + 1 > nrows or ofri + 1 > ncols:
            raise IndexError

        # Current Grid is enough, just return the requested values, and size_neighborhood = 9
        return grid[ofup : ofdo + 1, ofle : ofri + 1], 9

    except IndexError:

        # Count the number of neighbours
        size_neighborhood = 9

        invariant = np.array(invariant, dtype=grid.dtype)

        # 1. Generate extended grid.

        # Grid lenght at step N.
        l = lambda n: 2 * n + 1
        ln = l(n)
        egrid = np.repeat(invariant, ln * ln).reshape(ln, ln)

        # 2. Populate middle cell.

        mid = ln // 2
        egrid[mid, mid] = grid[row, col]

        is_legal = {
            "up": ofup >= 0,
            "down": ofdo <= nrows - 1,
            "left": ofle >= 0,
            "right": ofri <= ncols - 1,
        }

        # Distance
        d = lambda a, b: abs(b - a)

        # 3. Populate Up-Left Corner
        if is_legal["up"] and is_legal["left"]:
            size_neighborhood -= 0 

            egrid[mid - n : mid + 1, mid - n : mid + 1] = grid[
                row - n : row + 1, col - n : col + 1
            ]

        elif not is_legal["up"] and not is_legal["left"]:  # Both ilegal
            size_neighborhood -= 5


            br = d(row, 0)
            bc = d(col, 0)
            egrid[mid - br : mid + 1, mid - bc : mid + 1] = grid[
                row - br : row + 1, col - bc : col + 1
            ]

        elif not is_legal["up"]:
            size_neighborhood -= 3 


            br = d(row, 0)  # Distance to the border
            egrid[mid - br : mid + 1, mid - n : mid + 1] = grid[
                row - br : row + 1, col - n : col + 1
            ]
        elif not is_legal["left"]:
            size_neighborhood -= 3

            bc = d(col, 0)
            egrid[mid - n : mid + 1, mid - bc : mid + 1] = grid[
                row - n : row + 1, col - bc : col + 1
            ]

        # 4. Populate Up-Right Corner
        if is_legal["up"] and is_legal["right"]:
            size_neighborhood -= 0

            egrid[mid - n : mid + 1, mid : mid + n + 1] = grid[
                row - n : row + 1, col : col + n + 1
            ]

        elif not is_legal["up"] and not is_legal["right"]:
            size_neighborhood -= 2

            br = d(row, 0)
            bc = d(col, ncols)
            egrid[mid - br : mid + 1, mid : mid + bc] = grid[
                row - br : row + 1, col : col + bc
            ]

        elif not is_legal["up"]:
            size_neighborhood -= 0

            br = d(row, 0)
            egrid[mid - br : mid + 1, mid : mid + n + 1] = grid[
                row - br : row + 1, col : col + n + 1
            ]

        elif not is_legal["right"]:
            size_neighborhood -= 3

            bc = d(col, ncols)
            egrid[mid - n : mid + 1, mid : mid + bc] = grid[
                row - n : row + 1, col : col + bc
            ]

        # 5. Populate Down-Left Corner
        if is_legal["down"] and is_legal["left"]:
            size_neighborhood -= 0

            egrid[mid : mid + n + 1, mid - n : mid + 1] = grid[
                row : row + n + 1, col - n : col + 1
            ]

        elif not is_legal["down"] and not is_legal["left"]:
            size_neighborhood -= 2

            br = d(row, nrows)
            bc = d(col, 0)
            egrid[mid : mid + br, mid - bc : mid + 1] = grid[
                row : row + br, col - bc : col + 1
            ]

        elif not is_legal["down"]:
            size_neighborhood -= 3

            br = d(row, nrows)
            egrid[mid : mid + br, mid - n : mid + 1] = grid[
                row : row + br, col - n : col + 1
            ]

        elif not is_legal["left"]:
            size_neighborhood -= 0

            bc = d(col, 0)
            egrid[mid : mid + n + 1, mid - bc : mid + 1] = grid[
                row : row + n + 1, col - bc : col + 1
            ]

        # 6. Populate Down-Right Corner
        if is_legal["down"] and is_legal["right"]:
            size_neighborhood -= 0

            egrid[mid : mid + n + 1, mid : mid + n + 1] = grid[
                row : row + n + 1, col : col + n + 1
            ]

        elif not is_legal["down"] and not is_legal["right"]:
            size_neighborhood += 1 

            br = d(row, nrows)
            bc = d(col, ncols)
            egrid[mid : mid + br, mid : mid + bc] = grid[row : row + br, col : col + bc]

        elif not is_legal["down"]:
            size_neighborhood -= 0

            br = d(row, nrows)
            egrid[mid : mid + br, mid : mid + n + 1] = grid[
                row : row + br, col : col + n + 1
            ]

        elif not is_legal["right"]:
            size_neighborhood -= 0

            bc = d(col, ncols)
            egrid[mid : mid + n + 1, mid : mid + bc] = grid[
                row : row + n + 1, col : col + bc
            ]


        return egrid, size_neighborhood


def vonNeumann_n(
     position: tuple, grid: np.ndarray, invariant: Union[int, np.ndarray] = 0
):
    """Gets the N von Neuman neighborhood at given postion."""

    #Fixing n=1
    n = 1

    row, col = position
    nrows, ncols = grid.shape

    # Target offsets from position.
    ofup, ofdo = row + np.array([-n, +n])
    ofle, ofri = col + np.array([-n, +n])

    try:

        if ofup < 0 or ofle < 0 or ofdo + 1 > nrows or ofri + 1 > ncols:
            raise IndexError

        # Current Grid is enough, just return the requested values, and size_neighborhood = 5
        neighborhood = [grid[ofup,col],grid[row,ofle], grid[row,col],grid[row,ofri],grid[ofdo,col]] 
        return neighborhood, 5

    except IndexError:

        # Count the number of neighbours
        is_legal = {
            "up": ofup >= 0,
            "down": ofdo <= nrows - 1,
            "left": ofle >= 0,
            "right": ofri <= ncols - 1,
        }

        #There are 8 options
        if not is_legal["up"]:
            if not is_legal["left"]:
                neighborhood = [0,0, grid[row,col],grid[row,ofri],grid[ofdo,col]] 
                size_neighborhood = 3
            elif not is_legal["right"]:
                neighborhood = [0, grid[row,ofle], grid[row,col],0,grid[ofdo,col]] 
                size_neighborhood = 3
            else: 
                neighborhood = [0,grid[row,ofle], grid[row,col],grid[row,ofri],grid[ofdo,col]] 
                size_neighborhood = 4
        elif not is_legal["down"]:
            if not is_legal["left"]:
                neighborhood = [grid[ofup,col],0,grid[row,col],grid[row,ofri],0] 
                size_neighborhood = 3
            elif not is_legal["right"]:
                neighborhood = [grid[ofup,col], grid[row,ofle], grid[row,col],0,0] 
                size_neighborhood = 3
            else: 
                neighborhood = [grid[ofup,col],grid[row,ofle], grid[row,col],grid[row,ofri],0] 
                size_neighborhood = 4
        else:
            if not is_legal["left"]:
                neighborhood = [grid[ofup,col],0,grid[row,col],grid[row,ofri],grid[ofdo,col]] 
                size_neighborhood = 4
            else: 
                neighborhood = [grid[ofup,col],grid[row,ofle], grid[row,col],0,grid[ofdo,col]] 
                size_neighborhood = 4

        return neighborhood, size_neighborhood



# Depracated: Still used as interface for CAs.
# Superseded by Moore N function.
def neighborhood_at(grid, pos, invariant=0):
    """
    Calculates the Moore's neighborhood of cell at target position 'pos'.
    The boundary conditions are invariant and set to 'empty'.
    Returns a named tuple with the values of the nighborhood cells in the following
    order: up_left, up, up_right,
            left, self, right,
            down_left, down, down_right
    """
    from collections import namedtuple

    Neighbors = namedtuple(
        "Neighbors",
        [
            "up_left",
            "up",
            "up_right",
            "left",
            "self",
            "right",
            "down_left",
            "down",
            "down_right",
        ],
    )

    N = 1
    neighborhood, size_neighborhood = moore_n(N, pos, grid, invariant)
    neighborhood = neighborhood.flatten().tolist()
    return Neighbors(*neighborhood), size_neighborhood

def neighborhood_vonNeumann(grid, pos, invariant=0):
    """
    Calculates the Moore's neighborhood of cell at target position 'pos'.
    The boundary conditions are invariant and set to 'empty'.
    Returns a named tuple with the values of the nighborhood cells in the following
    order: up_left, up, up_right,
            left, self, right,
            down_left, down, down_right
    """
    from collections import namedtuple

    Neighbors = namedtuple(
        "Neighbors",
        [
            "up",
            "left",
            "self",
            "right",
            "down",
        ],
    )

    neighborhood, size_neighborhood = vonNeumann_n(pos, grid, invariant)
    return Neighbors(*neighborhood), size_neighborhood