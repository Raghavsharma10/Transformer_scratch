def draw_layer(ax, layer):
    """Draws a layer on the given matplotlib axis.

    Args:
        ax (axis): the matplotlib axis to draw on
        layer (layer): the layers to plot
    """
    ax.set_aspect('equal', 'datalim')
    ax.plot(*layer)
    ax.axis('off')