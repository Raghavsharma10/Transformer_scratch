def spht(ssphere, nmax=None, mmax=None):
    """Transforms ScalarPatternUniform object *ssphere* into a set of scalar
    spherical harmonics stored in ScalarCoefs.

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

    if nmax == None:
        nmax = ssphere.nrows - 2 
        mmax = int(ssphere.ncols / 2) - 1
    elif mmax == None:
        mmax = nmax

    if mmax > nmax:
        raise ValueError(err_msg['nmax_g_mmax'])

    if nmax >= ssphere.nrows - 1:
        raise ValueError(err_msg['nmax_too_lrg'])

    if mmax >= ssphere.ncols / 2:
        raise ValueError(err_msg['mmax_too_lrg'])

    dnrows = ssphere._dsphere.shape[0]
    ncols = ssphere._dsphere.shape[1]

    if np.mod(ncols, 2) == 1:
        raise ValueError(err_msg['ncols_even'])

    fdata = np.fft.fft2(ssphere._dsphere) / (dnrows * ncols)
    ops.fix_even_row_data_fc(fdata)
    
    fdata_extended = np.zeros([dnrows + 2, ncols], dtype=np.complex128)

    ops.pad_rows_fdata(fdata, fdata_extended)

    ops.sin_fc(fdata_extended)
    
    N = nmax + 1;
    NC = N + mmax * (2 * N - mmax - 1);
    sc = np.zeros(NC, dtype=np.complex128)
    # check if we are using c extended versions of the code or not
    if use_cext: 
        csphi.fc_to_sc(fdata_extended, sc, nmax, mmax)
    else:   
        sc = pysphi.fc_to_sc(fdata_extended, nmax, mmax)
                                
    return ScalarCoefs(sc, nmax, mmax)