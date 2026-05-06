def std_ratio(sim=None, obs=None, node=None, skip_nan=False):
    """Calculate the ratio between the standard deviation of the simulated
    and the observed values.

    >>> from hydpy import round_
    >>> from hydpy import std_ratio
    >>> round_(std_ratio(sim=[1.0, 2.0, 3.0], obs=[1.0, 2.0, 3.0]))
    0.0
    >>> round_(std_ratio(sim=[1.0, 1.0, 1.0], obs=[1.0, 2.0, 3.0]))
    -1.0
    >>> round_(std_ratio(sim=[0.0, 3.0, 6.0], obs=[1.0, 2.0, 3.0]))
    2.0

    See the documentation on function |prepare_arrays| for some
    additional instructions for use of function |std_ratio|.
    """
    sim, obs = prepare_arrays(sim, obs, node, skip_nan)
    return numpy.std(sim)/numpy.std(obs)-1.