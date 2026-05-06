def aptknt(tau, order):
    """Create an acceptable knot vector.

Minimal emulation of MATLAB's ``aptknt``.

The returned knot vector can be used to generate splines of desired `order`
that are suitable for interpolation to the collocation sites `tau`.

Note that this is only possible when ``len(tau)`` >= `order` + 1.

When this condition does not hold, a valid knot vector is returned,
but using it to generate a spline basis will not have the desired effect
(the spline will return a length-zero array upon evaluation).

Parameters:
    tau:
        Python list or rank-1 array, collocation sites

    order:
        int, >= 0, order of spline

Returns:
    rank-1 array, `k` copies of ``tau[0]``, then ``aveknt(tau[1:-1], k-1)``,
    and finally `k` copies of ``tau[-1]``, where ``k = min(order+1, len(tau))``.
"""
    tau = np.atleast_1d(tau)
    k   = order + 1

    if tau.ndim > 1:
        raise ValueError("tau must be a list or a rank-1 array")

    # emulate MATLAB behavior for the "k" parameter
    #
    # See
    #   https://se.mathworks.com/help/curvefit/aptknt.html
    #
    if len(tau) < k:
        k = len(tau)

    if not (tau == sorted(tau)).all():
        raise ValueError("tau must be nondecreasing")

    # last processed element needs to be:
    #     i + k - 1 = len(tau)- 1
    # =>  i + k = len(tau)
    # =>  i = len(tau) - k
    #
    u = len(tau) - k
    for i in range(u):
        if tau[i+k-1] == tau[i]:
            raise ValueError("k-fold (or higher) repeated sites not allowed, but tau[i+k-1] == tau[i] for i = %d, k = %d" % (i,k))

    # form the output sequence
    #
    prefix = [ tau[0]  ] * k
    suffix = [ tau[-1] ] * k

    # https://se.mathworks.com/help/curvefit/aveknt.html
    # MATLAB's aveknt():
    #  - averages successive k-1 entries, but ours averages k
    #  - seems to ignore the endpoints
    #
    tmp    = aveknt(tau[1:-1], k-1)
    middle = tmp.tolist()
    return np.array( prefix + middle + suffix, dtype=tmp.dtype )