def fill_value(dtype):
    '''Get a fill-value for a given dtype

    Parameters
    ----------
    dtype : type

    Returns
    -------
    `np.nan` if `dtype` is real or complex

    0 otherwise
    '''
    if np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.complexfloating):
        return dtype(np.nan)

    return dtype(0)