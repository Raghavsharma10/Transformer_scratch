def step(colors, x, controls=None):
    x = x.ravel()
    """Step interpolation from a set of colors. x belongs in [0, 1]."""
    assert (controls[0], controls[-1]) == (0., 1.)
    ncolors = len(colors)
    assert ncolors == len(controls) - 1
    assert ncolors >= 2
    x_step = _find_controls(x, controls, ncolors-1)
    return colors[x_step, ...]