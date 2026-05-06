def empirical_confidence_interval(sample, interval=0.95):
    """
    Compute specified symmetric confidence interval for empirical sample.

    Parameters
    ----------
    sample : numpy.array
        The empirical samples.
    interval : float, optional, default=0.95
        Size of desired symmetric confidence interval (0 < interval < 1)
        e.g. 0.68 for 68% confidence interval, 0.95 for 95% confidence interval

    Returns
    -------
    low : float
        The lower bound of the symmetric confidence interval.
    high : float
        The upper bound of the symmetric confidence interval.

    Examples
    --------
    >>> sample = np.random.randn(1000)
    >>> [low, high] = empirical_confidence_interval(sample)

    >>> [low, high] = empirical_confidence_interval(sample, interval=0.65)

    >>> [low, high] = empirical_confidence_interval(sample, interval=0.99)

    """
    # Sort sample in increasing order.
    sample = np.sort(sample)

    # Determine sample size.
    N = len(sample)

    # Compute low and high indices.
    low_index = int(np.round((N-1) * (0.5 - interval/2))) + 1
    high_index = int(np.round((N-1) * (0.5 + interval/2))) + 1

    # Compute low and high.
    low = sample[low_index]
    high = sample[high_index]

    return [low, high]