def fixed_poch(a, n):
    """Implementation of the Pochhammer symbol :math:`(a)_n` which handles negative integer arguments properly.
    
    Need conditional statement because scipy's impelementation of the Pochhammer
    symbol is wrong for negative integer arguments. This function uses the
    definition from
    http://functions.wolfram.com/GammaBetaErf/Pochhammer/02/
    
    Parameters
    ----------
    a : float
        The argument.
    n : nonnegative int
        The order.
    """
    # Old form, calls gamma function:
    # if a < 0.0 and a % 1 == 0 and n <= -a:
    #     p = (-1.0)**n * scipy.misc.factorial(-a) / scipy.misc.factorial(-a - n)
    # else:
    #     p = scipy.special.poch(a, n)
    # return p
    if (int(n) != n) or (n < 0):
        raise ValueError("Parameter n must be a nonnegative int!")
    n = int(n)
    # Direct form based on product:
    terms = [a + k for k in range(0, n)]
    return scipy.prod(terms)