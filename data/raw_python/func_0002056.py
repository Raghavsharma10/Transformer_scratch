def mad(arr, relative=True):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med = np.nanmedian(arr, axis=1)
        mad = np.nanmedian(np.abs(arr - med[:, np.newaxis]), axis=1)
        if relative:
            return mad / med
        else:
            return mad