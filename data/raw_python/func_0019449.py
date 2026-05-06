def get_covariance(self, chain=0, parameters=None):
        """
        Takes a chain and returns the covariance between chain parameters.

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
                2D covariance matrix.
        """
        index = self.parent._get_chain(chain)
        assert len(index) == 1, "Please specify only one chain, have %d chains" % len(index)
        chain = self.parent.chains[index[0]]
        if parameters is None:
            parameters = chain.parameters

        data = chain.get_data(parameters)
        cov = np.atleast_2d(np.cov(data, aweights=chain.weights, rowvar=False))

        return parameters, cov