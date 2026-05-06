def mod2pi(ts):
    """For a timeseries where all variables represent phases (in radians),
    return an equivalent timeseries where all values are in the range (-pi, pi]
    """
    return np.pi - np.mod(np.pi - ts, 2*np.pi)