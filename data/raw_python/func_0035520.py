def _update_dPrxy(self):
        """Update `dPrxy`, accounting for dependence of `Prxy` on `omega2`."""
        super(ExpCM_empirical_phi_divpressure, self)._update_dPrxy()
        if 'omega2' in self.freeparams:
            with scipy.errstate(divide='raise', under='raise', over='raise',
                            invalid='ignore'):
                scipy.copyto(self.dPrxy['omega2'], -self.ln_piAx_piAy_beta
                        * self.Qxy * self.omega /
                        (1 - self.piAx_piAy_beta), where=CODON_NONSYN)
            scipy.copyto(self.dPrxy['omega2'], self.Qxy * self.omega,
                       where=scipy.logical_and(CODON_NONSYN, scipy.fabs(1 -
                       self.piAx_piAy_beta) < ALMOST_ZERO))
            for r in range(self.nsites):
                self.dPrxy['omega2'][r] *= self.deltar[r]
            _fill_diagonals(self.dPrxy['omega2'], self._diag_indices)