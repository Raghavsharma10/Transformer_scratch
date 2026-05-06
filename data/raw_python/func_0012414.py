def log_subtract(loga, logb):
    r"""Numerically stable method for avoiding overflow errors when calculating
    :math:`\log (a-b)`, given :math:`\log (a)`, :math:`\log (a)` and that
    :math:`a > b`.

    See https://hips.seas.harvard.edu/blog/2013/01/09/computing-log-sum-exp/
    for more details.

    Parameters
    ----------
    loga: float
    logb: float
        Must be less than loga.

    Returns
    -------
    log(a - b): float
    """
    return loga + np.log(1 - np.exp(logb - loga))