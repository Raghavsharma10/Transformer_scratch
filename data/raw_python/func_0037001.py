def spht_slow(ssphere, nmax, mmax):
    """(PURE PYTHON) Transforms ScalarPatternUniform object *ssphere* 
    into a set of scalar spherical harmonics stored in ScalarCoefs.

    Example::

        >>> p = spherepy.random_patt_uniform(6, 8)
        >>> c = spherepy.spht(p)
        >>> spherepy.pretty_coefs(c)

    Args:
      ssphere (ScalarPatternUniform): The pattern to be transformed.

      nmax (int, optional): The maximum number of *n* values required. If a 
      value isn't passed, *nmax* is the number of rows in ssphere minus one.

      mmax (int, optional): The maximum number of *m* values required. If a 
      value isn't passed, *mmax* is half the number of columns in ssphere
      minus one.

    Returns:
      ScalarCoefs: The object containing the coefficients of the scalar
      spherical harmonic transform.


    Raises:
      ValueError: If *nmax* and *mmax* are too large or *mmax* > *nmax*.

    """

    if mmax > nmax:
        raise ValueError(err_msg['nmax_g_mmax'])

    nrows = ssphere._dsphere.shape[0]
    ncols = ssphere._dsphere.shape[1]

    if np.mod(nrows, 2) == 1 or np.mod(ncols, 2) == 1:
        raise ValueError(err_msg['ncols_even'])

    fdata = np.fft.fft2(ssphere._dsphere) / (nrows * ncols)
    ops.fix_even_row_data_fc(fdata)
    
    fdata_extended = np.zeros([nrows + 2, ncols], dtype=np.complex128)

    ops.pad_rows_fdata(fdata, fdata_extended)

    ops.sin_fc(fdata_extended)
    
    sc = pysphi.fc_to_sc(fdata_extended, nmax, mmax)
                                
    return ScalarCoefs(sc, nmax, mmax)