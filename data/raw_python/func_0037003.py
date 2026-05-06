def ispht_slow(scoefs, nrows, ncols):
    """(PURE PYTHON) Transforms ScalarCoefs object *scoefs* into a scalar
    pattern ScalarPatternUniform.

    Example::

        >>> c = spherepy.random_coefs(3,3)
        >>> p = spherepy.ispht(c)
        >>> print(p)

    Args:
      scoefs (ScalarCoefs): The coefficients to be transformed to pattern
      space.

      nrows (int): The number of rows desired in the pattern.

      ncols (int): The number of columns desired in the pattern. This must be 
      an even number.

    Returns:
      ScalarPatternUniform: This is the pattern. It contains a NumPy array that
      can be viewed with *patt.cdata*.


    Raises:
      ValueError: Is raised if *ncols* isn't even.

    """

    dnrows = 2 * nrows - 2

    if np.mod(ncols, 2) == 1:
        raise ValueError(err_msg['ncols_even'])

    fdata = pysphi.sc_to_fc(scoefs._vec,
                            scoefs._nmax,
                            scoefs._mmax,
                            dnrows, ncols)
    
    ds = np.fft.ifft2(fdata) * dnrows * ncols

    return ScalarPatternUniform(ds, doublesphere=True)