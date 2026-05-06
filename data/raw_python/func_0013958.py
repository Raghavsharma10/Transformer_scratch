def nan_circmean(samples, high=2.0*np.pi, low=0.0, axis=None):
    """NaN insensitive version of scipy's circular mean routine

    Parameters
    -----------
    samples : array_like
        Input array
    low : float or int
        Lower boundary for circular standard deviation range (default=0)
    high: float or int
        Upper boundary for circular standard deviation range (default=2 pi)
    axis : int or NoneType
        Axis along which standard deviations are computed.  The default is to
        compute the standard deviation of the flattened array

    Returns
    --------
    circmean : float
        Circular mean
    """

    samples = np.asarray(samples)
    samples = samples[~np.isnan(samples)]
    if samples.size == 0:
        return np.nan

    # Ensure the samples are in radians
    ang = (samples - low) * 2.0 * np.pi / (high - low)

    # Calculate the means of the sine and cosine, as well as the length
    # of their unit vector
    ssum = np.sin(ang).sum(axis=axis)
    csum = np.cos(ang).sum(axis=axis)
    res = np.arctan2(ssum, csum)

    # Bring the range of the result between 0 and 2 pi
    mask = res < 0.0

    if mask.ndim > 0:
        res[mask] += 2.0 * np.pi
    elif mask:
        res += 2.0 * np.pi

    # Calculate the circular standard deviation
    circmean = res * (high - low) / (2.0 * np.pi) + low
    return circmean