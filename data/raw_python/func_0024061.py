def linear_warp(X, d, n, *args):
    r"""Warp inputs with a linear transformation.
    
    Applies the warping
    
    .. math::
        
        w(x) = \frac{x-a}{b-a}
    
    to each dimension. If you set `a=min(X)` and `b=max(X)` then this is a
    convenient way to map your inputs to the unit hypercube.
    
    Parameters
    ----------
    X : array, (`M`,)
        `M` inputs from dimension `d`.
    d : non-negative int
        The index (starting from zero) of the dimension to apply the warping to.
    n : non-negative int
        The derivative order to compute.
    *args : 2N scalars
        The remaining parameters to describe the warping, given as scalars.
        These are given as `a_i`, `b_i` for each of the `D` dimensions. Note
        that these must ALL be provided for each call.
    """
    X = scipy.asarray(X, dtype=float)
    
    a = args[2 * d]
    b = args[2 * d + 1]
    
    if n == 0:
        return (X - a) / (b - a)
    elif n == 1:
        return 1.0 / (b - a) * scipy.ones_like(X)
    else:
        return scipy.zeros_like(X)