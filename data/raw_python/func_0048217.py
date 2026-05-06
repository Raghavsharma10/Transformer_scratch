def set_axis_color(axis, color, alpha=None):
    """Sets the spine color of all sides of an axis (top, right, bottom, left)."""
    for side in ('top', 'right',  'bottom', 'left'):
        spine = axis.spines[side]
        spine.set_color(color)
        if alpha is not None:
            spine.set_alpha(alpha)