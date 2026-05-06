def nse(sim=None, obs=None, node=None, skip_nan=False):
    """Calculate the efficiency criteria after Nash & Sutcliffe.

    If the simulated values predict the observed values as well
    as the average observed value (regarding the the mean square
    error), the NSE value is zero:

    >>> from hydpy import nse
    >>> nse(sim=[2.0, 2.0, 2.0], obs=[1.0, 2.0, 3.0])
    0.0
    >>> nse(sim=[0.0, 2.0, 4.0], obs=[1.0, 2.0, 3.0])
    0.0

    For worse and better simulated values the NSE is negative
    or positive, respectively:

    >>> nse(sim=[3.0, 2.0, 1.0], obs=[1.0, 2.0, 3.0])
    -3.0
    >>> nse(sim=[1.0, 2.0, 2.0], obs=[1.0, 2.0, 3.0])
    0.5

    The highest possible value is one:

    >>> nse(sim=[1.0, 2.0, 3.0], obs=[1.0, 2.0, 3.0])
    1.0

    See the documentation on function |prepare_arrays| for some
    additional instructions for use of function |nse|.
    """
    sim, obs = prepare_arrays(sim, obs, node, skip_nan)
    return 1.-numpy.sum((sim-obs)**2)/numpy.sum((obs-numpy.mean(obs))**2)