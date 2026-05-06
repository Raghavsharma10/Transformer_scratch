def _updateInternals(self):
        """Update internal attributes related to likelihood.

        Should be called any time branch lengths or model parameters
        are changed.
        """
        rootnode = self.nnodes - 1
        if self._distributionmodel:
            catweights = self.model.catweights
        else:
            catweights = scipy.ones(1, dtype='float')
        # When there are multiple categories, it is acceptable
        # for some (but not all) of them to have underflow at
        # any given site. Note that we still include a check for
        # Underflow by ensuring that none of the site likelihoods is
        # zero.
        undererrstate = 'ignore' if len(catweights) > 1 else 'raise'
        with scipy.errstate(over='raise', under=undererrstate,
                divide='raise', invalid='raise'):
            self.underflowlogscale.fill(0.0)
            self._computePartialLikelihoods()
            sitelik = scipy.zeros(self.nsites, dtype='float')
            assert (self.L[rootnode] >= 0).all(), str(self.L[rootnode])
            for k in self._catindices:
                sitelik += scipy.sum(self._stationarystate(k) *
                        self.L[rootnode][k], axis=1) * catweights[k]
            assert (sitelik > 0).all(), "Underflow:\n{0}\n{1}".format(
                    sitelik, self.underflowlogscale)
            self.siteloglik = scipy.log(sitelik) + self.underflowlogscale
            self.loglik = scipy.sum(self.siteloglik) + self.model.logprior
            if self.dparamscurrent:
                self._dloglik = {}
                for param in self.model.freeparams:
                    if self._distributionmodel and (param in
                            self.model.distributionparams):
                        name = self.model.distributedparam
                        weighted_dk = (self.model.d_distributionparams[param]
                                * catweights)
                    else:
                        name = param
                        weighted_dk = catweights
                    dsiteloglik = 0
                    for k in self._catindices:
                        dsiteloglik += (scipy.sum(
                                self._dstationarystate(k, name) *
                                self.L[rootnode][k] + self.dL[name][rootnode][k] *
                                self._stationarystate(k), axis=-1) *
                                weighted_dk[k])
                    dsiteloglik /= sitelik
                    self._dloglik[param] = (scipy.sum(dsiteloglik, axis=-1)
                            + self.model.dlogprior(param))
            if self.dtcurrent:
                self._dloglik_dt = 0
                dLnroot_dt = scipy.array([self.dL_dt[n2][rootnode] for
                        n2 in sorted(self.dL_dt.keys())])
                for k in self._catindices:
                    if isinstance(k, int):
                        dLnrootk_dt = dLnroot_dt.swapaxes(0, 1)[k]
                    else:
                        assert k == slice(None)
                        dLnrootk_dt = dLnroot_dt
                    self._dloglik_dt += catweights[k] * scipy.sum(
                            self._stationarystate(k) *
                            dLnrootk_dt, axis=-1)
                self._dloglik_dt /= sitelik
                self._dloglik_dt = scipy.sum(self._dloglik_dt, axis=-1)
                assert self._dloglik_dt.shape == self.t.shape