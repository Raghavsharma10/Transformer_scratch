def check_delta(fun, x, dxs, period=None):
    """Check the difference between two function values using the analytical gradient

       Arguments:
        | ``fun``  --  The function to be tested, more info below.
        | ``x``  --  The argument vector.
        | ``dxs``  --  A matrix where each row is a vector of small differences
                       to be added to the argument vector.

       Optional argument:
        | ``period``  --  If the function value is periodic, one may provide the
                          period such that differences are computed using
                          periodic boundary conditions.

       The function ``fun`` takes a mandatory argument ``x`` and an optional
       argument ``do_gradient``:
        | ``x``  --  The arguments of the function to be tested.
        | ``do_gradient``  --  When False, only the function value is returned.
                               When True, a 2-tuple with the function value and
                               the gradient are returned. [default=False]

       For every row in dxs, the following computation is repeated:

       1) D1 = 'f(x+dx) - f(x)' is computed.
       2) D2 = '0.5 (grad f(x+dx) + grad f(x)) . dx' is computed.

       A threshold is set to the median of the D1 set. For each case where |D1|
       is larger than the threshold, |D1 - D2|, should be smaller than the
       threshold.
    """
    dn1s = []
    dn2s = []
    dnds = []
    for dx in dxs:
        f0, grad0 = fun(x, do_gradient=True)
        f1, grad1 = fun(x+dx, do_gradient=True)
        grad = 0.5*(grad0+grad1)
        d1 = f1 - f0
        if period is not None:
            d1 -= np.floor(d1/period + 0.5)*period
        if hasattr(d1, '__iter__'):
            norm = np.linalg.norm
        else:
            norm = abs
        d2 = np.dot(grad, dx)

        dn1s.append(norm(d1))
        dn2s.append(norm(d2))
        dnds.append(norm(d1-d2))
    dn1s = np.array(dn1s)
    dn2s = np.array(dn2s)
    dnds = np.array(dnds)

    # Get the threshold (and mask)
    threshold = np.median(dn1s)
    mask = dn1s > threshold
    # Make sure that all cases for which dn1 is above the treshold, dnd is below
    # the threshold
    if not (dnds[mask] < threshold).all():
        raise AssertionError((
            'The first order approximation on the difference is too wrong. The '
            'threshold is %.1e.\n\nDifferences:\n%s\n\nFirst order '
            'approximation to differences:\n%s\n\nAbsolute errors:\n%s')
            % (threshold,
            ' '.join('%.1e' % v for v in dn1s[mask]),
            ' '.join('%.1e' % v for v in dn2s[mask]),
            ' '.join('%.1e' % v for v in dnds[mask])
        ))