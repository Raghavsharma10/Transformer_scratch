def timescales(self):
        r""" Relaxation timescales of the hidden transition matrix

        Returns
        -------
        ts : ndarray(m)
            relaxation timescales in units of the input trajectory time step,
            defined by :math:`-tau / ln | \lambda_i |, i = 2,...,nstates`, where
            :math:`\lambda_i` are the hidden transition matrix eigenvalues.

        """
        from msmtools.analysis.dense.decomposition import timescales_from_eigenvalues as _timescales

        self._ensure_spectral_decomposition()
        ts = _timescales(self._eigenvalues, tau=self._lag)
        return ts[1:]