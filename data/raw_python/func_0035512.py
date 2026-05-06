def _update_pi_vars(self):
        """Update variables that depend on `pi` from `zeta`.

        The updated variables are: `pi`, `pi_codon`, `ln_pi_codon`, `piAx_piAy`,
        `piAx_piAy_beta`, `ln_piAx_piAy_beta`, and `_logprior`.

        If `zeta` is undefined (as it will be on the first call), then create
        `zeta` and `origpi` from `pi` and `origbeta`."""
        minpi = self.PARAMLIMITS['pi'][0]
        if not hasattr(self, 'zeta'):
            # should only execute on first call to initialize zeta
            assert not hasattr(self, 'origpi')
            self.origpi = self.pi**self._origbeta
            for r in range(self.nsites):
                self.origpi[r] /= self.origpi[r].sum()
                self.origpi[r][self.origpi[r] < 2 * minpi] = 2 * minpi
                self.origpi[r] /= self.origpi[r].sum()
            self.pi = self.origpi.copy()
            self.zeta = scipy.ndarray(self.nsites * (N_AA - 1), dtype='float')
            self.tildeFrxy = scipy.zeros((self.nsites, N_CODON, N_CODON),
                    dtype='float')
            for r in range(self.nsites):
                zetaprod = 1.0
                for i in range(N_AA - 1):
                    zetari = 1.0 - self.pi[r][i] / zetaprod
                    self.zeta.reshape(self.nsites, N_AA - 1)[r][i] = zetari
                    zetaprod *= zetari
            (minzeta, maxzeta) = self.PARAMLIMITS['zeta']
            self.zeta[self.zeta < minzeta] = minzeta
            self.zeta[self.zeta > maxzeta] = maxzeta
            _checkParam('zeta', self.zeta, self.PARAMLIMITS, self.PARAMTYPES)
        else:
            # after first call, we are updating pi from zeta
            for r in range(self.nsites):
                zetaprod = 1.0
                for i in range(N_AA - 1):
                    zetari = self.zeta.reshape(self.nsites, N_AA - 1)[r][i]
                    self.pi[r][i] = zetaprod * (1 - zetari)
                    zetaprod *= zetari
                self.pi[r][N_AA - 1] = zetaprod
                self.pi[r][self.pi[r] < minpi] = minpi
                self.pi[r] /= self.pi[r].sum()

        super(ExpCM_fitprefs, self)._update_pi_vars()

        with scipy.errstate(divide='raise', under='raise', over='raise',
                invalid='ignore'):
            scipy.copyto(self.tildeFrxy, self.omega * self.beta *
                    (self.piAx_piAy_beta * (self.ln_piAx_piAy_beta - 1)
                    + 1) / (1 - self.piAx_piAy_beta)**2,
                    where=CODON_NONSYN)
        scipy.copyto(self.tildeFrxy, self.omega * self.beta / 2.0,
                where=scipy.logical_and(CODON_NONSYN, scipy.fabs(1 -
                self.piAx_piAy_beta) < ALMOST_ZERO))

        self._logprior = 0.0
        self._dlogprior = dict([(param, 0.0) for param in self.freeparams])
        if self.prior is None:
            pass
        elif self.prior[0] == 'invquadratic':
            (priorstr, c1, c2) = self.prior
            self._dlogprior = dict([(param, 0.0) for param in self.freeparams])
            self._dlogprior['zeta'] = scipy.zeros(self.zeta.shape, dtype='float')
            j = 0
            aaindex = scipy.arange(N_AA)
            for r in range(self.nsites):
                pidiffr = self.pi[r] - self.origpi[r]
                rlogprior = -c2 * scipy.log(1 + c1 * pidiffr**2).sum()
                self._logprior += rlogprior
                for i in range(N_AA - 1):
                    zetari = self.zeta[j]
                    self._dlogprior['zeta'][j] = -2 * c1 * c2 * (
                            pidiffr[i : ] / (1 + c1 * pidiffr[i : ]**2) *
                            self.pi[r][i : ] / (zetari - (aaindex == i).astype(
                            'float')[i : ])).sum()
                    j += 1
        else:
            raise ValueError("Invalid prior: {0}".format(self.prior))