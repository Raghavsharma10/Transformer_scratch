def dic(self):
        r""" Returns the corrected Deviance Information Criterion (DIC) for all chains loaded into ChainConsumer.

        If a chain does not have a posterior, this method will return `None` for that chain. **Note that
        the DIC metric is only valid on posterior surfaces which closely resemble multivariate normals!**
        Formally, we follow Liddle (2007) and first define *Bayesian complexity* as

        .. math::
            p_D = \bar{D}(\theta) - D(\bar{\theta}),

        where :math:`D(\theta) = -2\ln(P(\theta)) + C` is the deviance, where :math:`P` is the posterior
        and :math:`C` a constant. From here the DIC is defined as

        .. math::
            DIC \equiv D(\bar{\theta}) + 2p_D = \bar{D}(\theta) + p_D.

        Returns
        -------
        list[float]
            A list of all the DIC values - one per chain, in the order in which the chains were added.

        References
        ----------
        [1] Andrew R. Liddle, "Information criteria for astrophysical model selection", MNRAS (2007)
        """
        dics = []
        dics_bool = []
        for i, chain in enumerate(self.parent.chains):
            p = chain.posterior
            if p is None:
                dics_bool.append(False)
                self._logger.warn("You need to set the posterior for chain %s to get the DIC" % chain.name)
            else:
                dics_bool.append(True)
                num_params = chain.chain.shape[1]
                means = np.array([np.average(chain.chain[:, ii], weights=chain.weights) for ii in range(num_params)])
                d = -2 * p
                d_of_mean = griddata(chain.chain, d, means, method='nearest')[0]
                mean_d = np.average(d, weights=chain.weights)
                p_d = mean_d - d_of_mean
                dic = mean_d + p_d
                dics.append(dic)
        if len(dics) > 0:
            dics -= np.min(dics)
        dics_fin = []
        i = 0
        for b in dics_bool:
            if not b:
                dics_fin.append(None)
            else:
                dics_fin.append(dics[i])
                i += 1
        return dics_fin