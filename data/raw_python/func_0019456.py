def divide_chain(self, chain=0):
        """
        Returns a ChainConsumer instance containing all the walks of a given chain
        as individual chains themselves.

        This method might be useful if, for example, your chain was made using
        MCMC with 4 walkers. To check the sampling of all 4 walkers agree, you could
        call this to get a ChainConsumer instance with one chain for ech of the
        four walks. If you then plot, hopefully all four contours
        you would see agree.

        Parameters
        ----------
        chain : int|str, optional
            The index or name of the chain you want divided

        Returns
        -------
        ChainConsumer
            A new ChainConsumer instance with the same settings as the parent instance, containing
            ``num_walker`` chains.
        """
        indexes = self._get_chain(chain)
        con = ChainConsumer()

        for index in indexes:
            chain = self.chains[index]
            assert chain.walkers is not None, "The chain you have selected was not added with any walkers!"
            num_walkers = chain.walkers
            data = np.split(chain.chain, num_walkers)
            ws = np.split(chain.weights, num_walkers)
            for j, (c, w) in enumerate(zip(data, ws)):
                con.add_chain(c, weights=w, name="Chain %d" % j, parameters=chain.parameters)
        return con