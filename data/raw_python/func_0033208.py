def bisect(func, a, b, xtol=1e-12, maxiter=100):
    """
    Finds the root of `func` using the bisection method.

    Requirements
    ------------
    - func must be continuous function that accepts a single number input
      and returns a single number
    - `func(a)` and `func(b)` must have opposite sign

    Parameters
    ----------
    func : function
        the function that we want to find the root of
    a : number
        one of the bounds on the input
    b : number
        the other bound on the input
    xtol : number, optional
        the solution tolerance of the input value. The algorithm is
        considered converged if `abs(b-a)2. < xtol`
    maxiter : number, optional
        the maximum number of iterations allowed for convergence
    """
    fa = func(a)
    if fa == 0.:
        return a

    fb = func(b)
    if fb == 0.:
        return b

    assert sign(fa) != sign(fb)

    for i in xrange(maxiter):
        c = (a + b) / 2.
        fc = func(c)
        if fc == 0. or abs(b - a) / 2. < xtol:
            return c
        if sign(fc) == sign(func(a)):
            a = c
        else:
            b = c
    else:
        raise RuntimeError('Failed to converge after %d iterations.' % maxiter)