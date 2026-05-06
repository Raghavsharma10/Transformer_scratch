def _check_inputs(z, m):
    """Check inputs are arrays of same length or array and a scalar."""
    try:
        nz = len(z)
        z = np.array(z)
    except TypeError:
        z = np.array([z])
        nz = len(z)
    try:
        nm = len(m)
        m = np.array(m)
    except TypeError:
        m = np.array([m])
        nm = len(m)

    if (z < 0).any() or (m < 0).any():
        raise ValueError('z and m must be positive')

    if nz != nm and nz > 1 and nm > 1:
        raise ValueError('z and m arrays must be either equal in length, \
                          OR of different length with one of length 1.')

    else:
        if type(z) != np.ndarray:
            z = np.array(z)
        if type(m) != np.ndarray:
            m = np.array(m)

        return z, m