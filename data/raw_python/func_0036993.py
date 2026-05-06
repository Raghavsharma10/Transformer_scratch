def zeros_coefs(nmax, mmax, coef_type=scalar):
    """Returns a ScalarCoefs object or a VectorCoeffs object where each of the 
    coefficients is set to 0. The structure is such that *nmax* is th largest 
    *n* can be in c[n, m], and *mmax* is the largest *m* can be for any *n*.
    (See *ScalarCoefs* and *VectorCoefs* for details.)
    

    Examples::
        
        >>> c = spherepy.zeros_coefs(5, 3, coef_type = spherepy.scalar)
        >>> c = spherepy.zeros_coefs(5, 3) # same as above
        >>> vc = spherepy.zeros_coefs(5, 3, coef_type = spherepy.vector)

    Args:
          nmax (int): Largest *n* value in the set of modes.

          mmax (int): Largest abs(*m*) value in the set of modes.

          coef_type (int, optional): Set to 0 for scalar, and 1 for vector.
          The default option is scalar. If you would like to return a set of 
          vector spherical hamonic coefficients, the preferred way to do so 
          is vc = spherepy.zeros_coefs( 10, 12, coef_type = spherepy.vector).

    Returns:
      coefs: Returns a ScalarCoefs object if coef_type is either blank or
      set to 0. Returns a VectorCoefs object if coef_type = 1.

    Raises:
      TypeError: If coef_type is anything but 0 or 1.

    """

    if(mmax > nmax):
        raise ValueError(err_msg['nmax_g_mmax'])

    if(coef_type == scalar):
        L = (nmax + 1) + mmax * (2 * nmax - mmax + 1)
        vec = np.zeros(L, dtype=np.complex128)
        return  ScalarCoefs(vec, nmax, mmax)
    elif(coef_type == vector):
        L = (nmax + 1) + mmax * (2 * nmax - mmax + 1)
        vec1 = np.zeros(L, dtype=np.complex128)
        vec2 = np.zeros(L, dtype=np.complex128)
        return  VectorCoefs(vec1, vec2, nmax, mmax)
    else:
        raise TypeError(err_msg['ukn_coef_t'])