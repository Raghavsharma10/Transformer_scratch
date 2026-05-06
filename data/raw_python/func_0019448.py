def get_correlations(self, chain=0, parameters=None):
        """
        Takes a chain and returns the correlation between chain parameters.

        Parameters
        ----------
        chain : int|str, optional
            The chain index or name. Defaults to first chain.
        parameters : list[str], optional
            The list of parameters to compute correlations. Defaults to all parameters
            for the given chain.

        Returns
        -------
            tuple
                The first index giving a list of parameter names, the second index being the
                2D correlation matrix.
        """
        parameters, cov = self.get_covariance(chain=chain, parameters=parameters)
        diag = np.sqrt(np.diag(cov))
        divisor = diag[None, :] * diag[:, None]
        correlations = cov / divisor
        return parameters, correlations