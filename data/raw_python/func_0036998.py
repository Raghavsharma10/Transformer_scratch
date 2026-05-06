def pretty_coefs(c):
    """Prints out the first 2 modes of a ScalarCoeffs object. This is mostly 
    used for instructional purposes.
    (*ScalarPatternUniform*)
    
    Example::
        
        >>> spherepy.pretty_coefs(c)
        
        c[n, m]
        =======

        2:       0j             0j            0j             0j             0j 
        1:                     1+0j           0j           -1+0j 
        0:                                    1j  
        n  -------------  -------------  -------------  -------------  -------------  
               m = -2         m = -1         m = 0          m = 1          m = 2    

    Args:
          c (ScalarCoefs): Coefficients to be printed.

    Returns:
      nothing, just outputs something pretty to the console.

    """

    cfit = c[0:2, :]
    cvec = cfit._vec

    sa = [_tiny_rep(val) for val in cvec]

    while len(sa) < 9:
        sa.append("")

    sa = [sa[n].center(13) for n in range(0, 9)]

    print(pretty_display_string.format(sa[0], sa[1], sa[2],
                                       sa[3], sa[4], sa[5],
                                       sa[6], sa[7], sa[8]))