def double_sphere(cdata, sym):
    """ Ensures that the data within cdata has double sphere symmetry.

    Example::

        >>> spherepy.doublesphere(cdata, 1)

    Args:
        sym (int): is 1 for scalar data and -1 for vector data

    Returns:
        numpy.array([*,*], dtype=np.complex128) containing array with 
        doublesphere symmetry.
    """
    
    nrows = cdata.shape[0]
    ncols = cdata.shape[1]

    ddata = np.zeros([nrows, ncols], dtype=np.complex128)

    for n in xrange(0, nrows):
        for m in xrange(0, ncols):
            s = sym * cdata[np.mod(nrows - n, nrows),
                          np.mod(int(np.floor(ncols / 2)) + m, ncols)]
            t = cdata[n, m]

            if s * t == 0:
                ddata[n, m] = s + t
            else:
                ddata[n, m] = (s + t) / 2

    return ddata