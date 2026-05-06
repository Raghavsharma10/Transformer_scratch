def aic(self):
        r""" Returns the corrected Akaike Information Criterion (AICc) for all chains loaded into ChainConsumer.

        If a chain does not have a posterior, number of data points, and number of free parameters
        loaded, this method will return `None` for that chain. Formally, the AIC is defined as

        .. math::
            AIC \equiv -2\ln(P) + 2k,

        where :math:`P` represents the posterior, and :math:`k` the number of model parameters. The AICc
        is then defined as

        .. math::
            AIC_c \equiv AIC + \frac{2k(k+1)}{N-k-1},

        where :math:`N` represents the number of independent data points used in the model fitting.
        The AICc is a correction for the AIC to take into account finite chain sizes.

        Returns
        -------
        list[float]
            A list of all the AICc values - one per chain, in the order in which the chains were added.
        """
        aics = []
        aics_bool = []
        for i, chain in enumerate(self.parent.chains):
            p, n_data, n_free = chain.posterior, chain.num_eff_data_points, chain.num_free_params
            if p is None or n_data is None or n_free is None:
                aics_bool.append(False)
                missing = ""
                if p is None:
                    missing += "posterior, "
                if n_data is None:
                    missing += "num_eff_data_points, "
                if n_free is None:
                    missing += "num_free_params, "

                self._logger.warn("You need to set %s for chain %s to get the AIC" %
                                  (missing[:-2], chain.name))
            else:
                aics_bool.append(True)
                c_cor = (1.0 * n_free * (n_free + 1) / (n_data - n_free - 1))
                aics.append(2.0 * (n_free + c_cor - np.max(p)))
        if len(aics) > 0:
            aics -= np.min(aics)
        aics_fin = []
        i = 0
        for b in aics_bool:
            if not b:
                aics_fin.append(None)
            else:
                aics_fin.append(aics[i])
                i += 1
        return aics_fin