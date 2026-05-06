def zeros_patt_uniform(nrows, ncols, patt_type=scalar):
    """Returns a ScalarPatternUniform object or a VectorPatternUniform object
    where each of the elements is set to 0. *nrows* is the number of rows in 
    the pattern, which corresponds to the theta axis. *ncols* must be even
    and is the number of columns in the pattern and corresponds to the phi 
    axis.
    (See *ScalarPatternUniform* and *VectorPatternUniform* for details.)
    
    Examples::
        
        >>> f = spherepy.zeros_patt_uniform(6, 8, coef_type = spherepy.scalar)
        >>> f = spherepy.zeros_patt_uniform(6, 8) # same as above
        >>> F = spherepy.zeros_patt_uniform(6, 8, coef_type = spherepy.vector)

    Args:
          nrows (int): Number of rows corresponding to the theta axis.

          ncols (int): Number of columns corresponding to the phi axis. To get 
          the speed and accuracy I need, this value **must** be even. 

          coef_type (int, optional): Set to 0 for scalar, and 1 for vector.
          The default option is scalar. 

    Returns:
      coefs: Returns a ScalarPatternUniform object if coef_type is either 
      blank or set to 0. Returns a VectorPatternUniform object if 
      coef_type = 1.

    Raises:
      ValueError: If ncols is not even.

      TypeError: If coef_type is anything but 0 or 1.

    """
    
    if np.mod(ncols, 2) == 1:
        raise ValueError(err_msg['ncols_even'])

    if(patt_type == scalar):
        cdata = np.zeros((2 * nrows - 2, ncols), dtype=np.complex128)
        return ScalarPatternUniform(cdata, doublesphere=True)

    elif(patt_type == vector):
        tcdata = np.zeros((2 * nrows - 2, ncols), dtype=np.complex128)
        pcdata = np.zeros((2 * nrows - 2, ncols), dtype=np.complex128)
        return TransversePatternUniform(tcdata, pcdata, doublesphere=True)

    else:
        raise TypeError(err_msg['ukn_patt_t'])