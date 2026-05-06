def smooth(data, fw):
    """Smooth data with a moving average."""
    if fw == 0:
        fdata = data
    else:
        fdata = lfilter(np.ones(fw)/fw, 1, data)
    return fdata