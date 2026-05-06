def geweke(self, chain=None, first=0.1, last=0.5, threshold=0.05):
        """ Runs the Geweke diagnostic on the supplied chains.

        Parameters
        ----------
        chain : int|str, optional
            Which chain to run the diagnostic on. By default, this is `None`,
            which will run the diagnostic on all chains. You can also
            supply and integer (the chain index) or a string, for the chain
            name (if you set one).
        first : float, optional
            The amount of the start of the chain to use
        last : float, optional
            The end amount of the chain to use
        threshold : float, optional
            The p-value to use when testing for normality.

        Returns
        -------
        float
            whether or not the chains pass the test

        """
        if chain is None:
            return np.all([self.geweke(k, threshold=threshold) for k in range(len(self.parent.chains))])

        index = self.parent._get_chain(chain)
        assert len(index) == 1, "Please specify only one chain, have %d chains" % len(index)
        chain = self.parent.chains[index[0]]

        num_walkers = chain.walkers
        assert num_walkers is not None and num_walkers > 0, \
            "You need to specify the number of walkers to use the Geweke diagnostic."
        name = chain.name
        data = chain.chain
        chains = np.split(data, num_walkers)
        n = 1.0 * chains[0].shape[0]
        n_start = int(np.floor(first * n))
        n_end = int(np.floor((1 - last) * n))
        mean_start = np.array([np.mean(c[:n_start, i])
                               for c in chains for i in range(c.shape[1])])
        var_start = np.array([self._spec(c[:n_start, i]) / c[:n_start, i].size
                              for c in chains for i in range(c.shape[1])])
        mean_end = np.array([np.mean(c[n_end:, i])
                             for c in chains for i in range(c.shape[1])])
        var_end = np.array([self._spec(c[n_end:, i]) / c[n_end:, i].size
                            for c in chains for i in range(c.shape[1])])
        zs = (mean_start - mean_end) / (np.sqrt(var_start + var_end))
        _, pvalue = normaltest(zs)
        print("Gweke Statistic for chain %s has p-value %e" % (name, pvalue))
        return pvalue > threshold