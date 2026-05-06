def collmat(self, tau, deriv_order=0):
        """Compute collocation matrix.

Parameters:
    tau:
        Python list or rank-1 array, collocation sites
    deriv_order:
        int, >=0, order of derivative for which to compute the collocation matrix.
        The default is 0, which means the function value itself.

Returns:
    A:
        if len(tau) > 1, rank-2 array such that
            A[i,j] = D**deriv_order B_j(tau[i])
        where
            D**k  = kth derivative (0 for function value itself)

        if len(tau) == 1, rank-1 array such that
            A[j]   = D**deriv_order B_j(tau)

Example:
    If the coefficients of a spline function are given in the vector c, then::

        np.sum( A*c, axis=-1 )

    will give a rank-1 array of function values at the sites tau[i] that were supplied
    to `collmat`.

    Similarly for derivatives (if the supplied `deriv_order`> 0).

"""
        # get number of basis functions and output dtype
        dummy = self.__call__(0.)
        nbasis = dummy.shape[0]

        tau = np.atleast_1d(tau)
        if tau.ndim > 1:
            raise ValueError("tau must be a list or a rank-1 array")

        A = np.empty( (tau.shape[0], nbasis), dtype=dummy.dtype )
        f = self.diff(order=deriv_order)
        for i,taui in enumerate(tau):
            A[i,:] = f(taui)

        return np.squeeze(A)