def incomplete_bell_poly(n, k, x):
    r"""Recursive evaluation of the incomplete Bell polynomial :math:`B_{n, k}(x)`.
    
    Evaluates the incomplete Bell polynomial :math:`B_{n, k}(x_1, x_2, \dots, x_{n-k+1})`,
    also known as the partial Bell polynomial or the Bell polynomial of the
    second kind. This polynomial is useful in the evaluation of (the univariate)
    Faa di Bruno's formula which generalizes the chain rule to higher order
    derivatives.
    
    The implementation here is based on the implementation in:
    :py:func:`sympy.functions.combinatorial.numbers.bell._bell_incomplete_poly`
    Following that function's documentation, the polynomial is computed
    according to the recurrence formula:
    
    .. math::
        
        B_{n, k}(x_1, x_2, \dots, x_{n-k+1}) = \sum_{m=1}^{n-k+1}x_m\binom{n-1}{m-1}B_{n-m, k-1}(x_1, x_2, \dots, x_{n-m-k})
        
    | The end cases are:
    | :math:`B_{0, 0} = 1`
    | :math:`B_{n, 0} = 0` for :math:`n \ge 1`
    | :math:`B_{0, k} = 0` for :math:`k \ge 1`
    
    Parameters
    ----------
    n : scalar int
        The first subscript of the polynomial.
    k : scalar int
        The second subscript of the polynomial.
    x : :py:class:`Array` of floats, (`p`, `n` - `k` + 1)
        `p` sets of `n` - `k` + 1 points to use as the arguments to
        :math:`B_{n,k}`. The second dimension can be longer than
        required, in which case the extra entries are silently ignored
        (this facilitates recursion without needing to subset the array `x`).
    
    Returns
    -------
    result : :py:class:`Array`, (`p`,)
        Incomplete Bell polynomial evaluated at the desired values.
    """
    if n == 0 and k == 0:
        return scipy.ones(x.shape[0], dtype=float)
    elif k == 0 and n >= 1:
        return scipy.zeros(x.shape[0], dtype=float)
    elif n == 0 and k >= 1:
        return scipy.zeros(x.shape[0], dtype=float)
    else:
        result = scipy.zeros(x.shape[0], dtype=float)
        for m in xrange(0, n - k + 1):
            result += x[:, m] * scipy.special.binom(n - 1, m) * incomplete_bell_poly(n - (m + 1), k - 1, x)
        return result