def compute_Kij(self, Xi, Xj, ni, nj, noise=False, hyper_deriv=None, k=None):
        r"""Compute covariance matrix between datasets `Xi` and `Xj`.
        
        Specify the orders of derivatives at each location with the `ni`, `nj`
        arrays. The `include_noise` flag is passed to the covariance kernel to
        indicate whether noise is to be included (i.e., for evaluation of
        :math:`K+\sigma I` versus :math:`K_*`).
        
        If `Xj` is None, the symmetric matrix :math:`K(X, X)` is formed.
        
        Note that type and dimension checking is NOT performed, as it is assumed
        the data are from inside the instance and have hence been sanitized by
        :py:meth:`add_data`.
        
        Parameters
        ----------
        Xi : array, (`M`, `D`)
            `M` input values of dimension `D`.
        Xj : array, (`P`, `D`)
            `P` input values of dimension `D`.
        ni : array, (`M`, `D`), non-negative integers
            `M` derivative orders with respect to the `Xi` coordinates.
        nj : array, (`P`, `D`), non-negative integers
            `P` derivative orders with respect to the `Xj` coordinates.
        noise : bool, optional
            If True, uses the noise kernel, otherwise uses the regular kernel.
            Default is False (use regular kernel).
        hyper_deriv : None or non-negative int, optional
            Index of the hyperparameter to compute the first derivative with
            respect to. If None, no derivatives are taken. Default is None (no
            hyperparameter derivatives).
        k : :py:class:`~gptools.kernel.core.Kernel` instance, optional
            The covariance kernel to used. Overrides `noise` if present.
                
        Returns
        -------
        Kij : array, (`M`, `P`)
            Covariance matrix between `Xi` and `Xj`.
        """
        if k is None:
            if not noise:
                k = self.k
            else:
                k = self.noise_k
        
        if Xj is None:
            symmetric = True
            Xj = Xi
            nj = ni
        else:
            symmetric = False
        
        # TODO: This technically doesn't take advantage of the symmetric case.
        # Might be worth trying to do that at some point, but this is vastly
        # superior to the double for loop implementation for which using
        # symmetry is easy.
        Xi_tile = scipy.repeat(Xi, Xj.shape[0], axis=0)
        ni_tile = scipy.repeat(ni, Xj.shape[0], axis=0)
        Xj_tile = scipy.tile(Xj, (Xi.shape[0], 1))
        nj_tile = scipy.tile(nj, (Xi.shape[0], 1))
        Kij = k(
            Xi_tile,
            Xj_tile,
            ni_tile,
            nj_tile,
            hyper_deriv=hyper_deriv,
            symmetric=symmetric
        )
        Kij = scipy.reshape(Kij, (Xi.shape[0], -1))
        
        return Kij