def corr(sim=None, obs=None, node=None, skip_nan=False):
    """Calculate the product-moment correlation coefficient after Pearson.

    >>> from hydpy import round_
    >>> from hydpy import corr
    >>> round_(corr(sim=[0.5, 1.0, 1.5], obs=[1.0, 2.0, 3.0]))
    1.0
    >>> round_(corr(sim=[4.0, 2.0, 0.0], obs=[1.0, 2.0, 3.0]))
    -1.0
    >>> round_(corr(sim=[1.0, 2.0, 1.0], obs=[1.0, 2.0, 3.0]))
    0.0

    See the documentation on function |prepare_arrays| for some
    additional instructions for use of function |corr|.
    """
    sim, obs = prepare_arrays(sim, obs, node, skip_nan)
    return numpy.corrcoef(sim, obs)[0, 1]