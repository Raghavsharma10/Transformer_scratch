def bic(self):
        r""" Returns the corrected Bayesian Information Criterion (BIC) for all chains loaded into ChainConsumer.

        If a chain does not have a posterior, number of data points, and number of free parameters
        loaded, this method will return `None` for that chain. Formally, the BIC is defined as

        .. math::
            BIC \equiv -2\ln(P) + k \ln(N),

        where :math:`P` represents the posterior, :math:`k` the number of model parameters and :math:`N`
        the number of independent data points used in the model fitting.

        Returns
        -------
        list[float]
            A list of all the BIC values - one per chain, in the order in which the chains were added.
        """
        bics = []
        bics_bool = []
        for i, chain in enumerate(self.parent.chains):
            p, n_data, n_free = chain.posterior, chain.num_eff_data_points, chain.num_free_params
            if p is None or n_data is None or n_free is None:
                bics_bool.append(False)
                missing = ""
                if p is None:
                    missing += "posterior, "
                if n_data is None:
                    missing += "num_eff_data_points, "
                if n_free is None:
                    missing += "num_free_params, "

                self._logger.warn("You need to set %s for chain %s to get the BIC" %
                                  (missing[:-2], chain.name))
            else:
                bics_bool.append(True)
                bics.append(n_free * np.log(n_data) - 2 * np.max(p))
        if len(bics) > 0:
            bics -= np.min(bics)
        bics_fin = []
        i = 0
        for b in bics_bool:
            if not b:
                bics_fin.append(None)
            else:
                bics_fin.append(bics[i])
                i += 1
        return bics_fin