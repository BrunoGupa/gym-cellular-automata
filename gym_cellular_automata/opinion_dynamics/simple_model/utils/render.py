from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import colors

from gym_cellular_automata.game_of_life.utils.render import TITLEFONT, parse_svg_into_mpl

from .config import CONFIG
from .helicopter_shape import SVG_PATH


NUM_OPINIONS     = CONFIG["num_opinions"]



DEFAULT_KWARGS = {
    "color_2": ("#2D2926FF","#E94B3CFF"),  
    "color_3": ("#F65058FF","#FBDE44FF","#28334AFF"),  
    "color_4": ("#322e2f", "#12a4d9", "#d9138a", "#e2d810"),
    "color_5": ("#AC92EB", "#4FC1E8", "#A0D568", "#FFCE54", "#ED5564"),
    "title_size": 184,#64,
    "title_color_2": "#2D2926FF",
    "title_color_3": "#28334AFF",
    "title_color_4": "#322e2f",
    "title_color_5": "#000000",  
    "helicopter_size": 21,
    "helicopter_color": "#DC143C",# Red "#FFFFFF",  # White
}


def plot_grid(grid, return_fig = False, **kwargs):

    kwargs = {**DEFAULT_KWARGS, **kwargs}


    color_mapping = cell_colors_to_cmap_and_norm(
        (kwargs["color_{}".format(NUM_OPINIONS)])
    )

    # Plot style
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(10,8))#15, 12))

    # Main Plot
    ax.imshow(
        grid, aspect="equal", cmap=color_mapping["cmap"], norm=color_mapping["norm"]
    )

    # Title
    fig.suptitle(
        "Influencer V0",
        color=kwargs["title_color_{}".format(NUM_OPINIONS)],
        font=TITLEFONT,
        fontsize=64,
        ha="center",
    )

    # Modify Ticks by Axes methods
    grid_ticks_settings(plt.gca(), n_row=grid.shape[0], n_col=grid.shape[1])
    
    if return_fig:
        return fig
    else:
        plt.show()


def add_helicopter(fig, pos, **kwargs):
    import matplotlib.patheffects as path_effects


    helicopter = parse_svg_into_mpl(SVG_PATH)
    pe = [path_effects.Stroke(linewidth=3, foreground="white"), path_effects.Normal()]
    kwargs = {**DEFAULT_KWARGS, **kwargs}

    ax = fig.get_axes()[0]
    row, col = pos

    ax.plot(
        col,
        row,
        marker=helicopter,
        markersize=kwargs["helicopter_size"],
        color=kwargs["helicopter_color"],
        fillstyle="none",
        path_effects=pe,
    )
    #return fig
    plt.show()


def cell_colors_to_cmap_and_norm(colors_environtment):
    """
    Mappings from Color to Cell Symbols.
    """
    symbols = 0, 1, 2, 3, 4, 5

    symbols_with_colors = zip(symbols, colors_environtment)
    symbols_with_colors_sorted_by_symbol = sorted(
        symbols_with_colors, key=itemgetter(0)
    )
    symbols, colors_environtment = zip(*symbols_with_colors_sorted_by_symbol)

    return {
        "cmap": colors.ListedColormap(colors_environtment),
        "norm": colors.BoundaryNorm([0, 1, 2, 3, 4], 4),
    }


def grid_ticks_settings(ax, n_row, n_col):

    # NO Labels for ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Major ticks
    ax.set_xticks(np.arange(0, n_col, 1))
    ax.set_yticks(np.arange(0, n_row, 1))

    # Minor ticks
    ax.set_xticks(np.arange(-0.5, n_col, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_row, 1), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which="minor", color="whitesmoke", linestyle="-", linewidth=2)
    ax.grid(which="major", color="w", linestyle="-", linewidth=0)
    ax.tick_params(axis="both", which="both", length=0)

    return ax
