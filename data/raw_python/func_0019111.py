def bias_abs(sim=None, obs=None, node=None, skip_nan=False):
    """Calculate the absolute difference between the means of the simulated
    and the observed values.

    >>> from hydpy import round_
    >>> from hydpy import bias_abs
    >>> round_(bias_abs(sim=[2.0, 2.0, 2.0], obs=[1.0, 2.0, 3.0]))
    0.0
    >>> round_(bias_abs(sim=[5.0, 2.0, 2.0], obs=[1.0, 2.0, 3.0]))
    1.0
    >>> round_(bias_abs(sim=[1.0, 1.0, 1.0], obs=[1.0, 2.0, 3.0]))
    -1.0

    See the documentation on function |prepare_arrays| for some
    additional instructions for use of function |bias_abs|.
    """
    sim, obs = prepare_arrays(sim, obs, node, skip_nan)
    return numpy.mean(sim-obs)