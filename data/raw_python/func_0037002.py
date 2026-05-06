def ispht(scoefs, nrows=None, ncols=None):
    """Transforms ScalarCoefs object *scoefs* into a scalar pattern 
    ScalarPatternUniform.

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

      ValueError: Is raised if *nrows* < *nmax* + 2 or *ncols* < 2 * *mmax* + 2.

    """

    if nrows == None:
        nrows = scoefs.nmax + 2 

    if ncols == None:
        ncols = 2 * scoefs.mmax + 2

    if nrows <= scoefs.nmax:
        raise ValueError(err_msg['inverse_terr'])

    if ncols < 2 * scoefs.mmax + 2:
        raise ValueError(err_msg['inverse_terr'])

    dnrows = int(2 * nrows - 2)

    if np.mod(ncols, 2) == 1:
        raise ValueError(err_msg['ncols_even'])

    if use_cext: 
        fdata = np.zeros([dnrows, ncols], dtype=np.complex128)
        csphi.sc_to_fc(fdata, scoefs._vec, scoefs._nmax, scoefs._mmax)
    else:   
        fdata = pysphi.sc_to_fc(scoefs._vec,
                            scoefs._nmax,
                            scoefs._mmax,
                            dnrows, ncols)
    
    ds = np.fft.ifft2(fdata) * dnrows * ncols

    return ScalarPatternUniform(ds, doublesphere=True)