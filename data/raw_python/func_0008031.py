def spcol(knots, order, tau):
    """Return collocation matrix.

Minimal emulation of MATLAB's ``spcol``.

Parameters:
    knots:
        rank-1 array, knot vector (with appropriately repeated endpoints; see `augknt`, `aptknt`)
    order:
        int, >= 0, order of spline
    tau:
        rank-1 array, collocation sites

Returns:
    rank-2 array A such that

        A[i,j] = D**{m(i)} B_j(tau[i])

    where
        m(i) = multiplicity of site tau[i]

        D**k  = kth derivative (0 for function value itself)
"""
    m = knt2mlt(tau)
    B = bspline.Bspline(knots, order)

    dummy = B(0.)
    nbasis = len(dummy)  # perform dummy evaluation to get number of basis functions

    A = np.empty( (tau.shape[0], nbasis), dtype=dummy.dtype )
    for i,item in enumerate(zip(tau,m)):
        taui,mi = item
        f       = B.diff(order=mi)
        A[i,:]  = f(taui)

    return A