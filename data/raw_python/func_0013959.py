def nan_circstd(samples, high=2.0*np.pi, low=0.0, axis=None):
    """NaN insensitive version of scipy's circular standard deviation routine

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
    circstd : float
        Circular standard deviation
    """

    samples = np.asarray(samples)
    samples = samples[~np.isnan(samples)]
    if samples.size == 0:
        return np.nan

    # Ensure the samples are in radians
    ang = (samples - low) * 2.0 * np.pi / (high - low)

    # Calculate the means of the sine and cosine, as well as the length
    # of their unit vector
    smean = np.sin(ang).mean(axis=axis)
    cmean = np.cos(ang).mean(axis=axis)
    rmean = np.sqrt(smean**2 + cmean**2)

    # Calculate the circular standard deviation
    circstd = (high - low) * np.sqrt(-2.0 * np.log(rmean)) / (2.0 * np.pi)
    return circstd