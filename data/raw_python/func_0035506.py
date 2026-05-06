def _update_dPrxy(self):
        """Update `dPrxy`."""
        if 'kappa' in self.freeparams:
            scipy.copyto(self.dPrxy['kappa'], self.Prxy / self.kappa,
                    where=CODON_TRANSITION)
            _fill_diagonals(self.dPrxy['kappa'], self._diag_indices)
        if 'omega' in self.freeparams:
            scipy.copyto(self.dPrxy['omega'], self.Frxy_no_omega * self.Qxy,
                    where=CODON_NONSYN)
            _fill_diagonals(self.dPrxy['omega'], self._diag_indices)
        if 'beta' in self.freeparams:
            self.dPrxy['beta'].fill(0)
            with scipy.errstate(divide='raise', under='raise', over='raise',
                    invalid='ignore'):
                scipy.copyto(self.dPrxy['beta'], self.Prxy *
                        (1 / self.beta + (self.piAx_piAy_beta *
                        (self.ln_piAx_piAy_beta / self.beta) /
                        (1 - self.piAx_piAy_beta))), where=CODON_NONSYN)
            scipy.copyto(self.dPrxy['beta'], self.Prxy/self.beta *
                    (1 - self.piAx_piAy_beta), where=scipy.logical_and(
                    CODON_NONSYN, scipy.fabs(1 - self.piAx_piAy_beta)
                    < ALMOST_ZERO))
            _fill_diagonals(self.dPrxy['beta'], self._diag_indices)
        if 'eta' in self.freeparams:
            for i in range(N_NT - 1):
                for w in range(i, N_NT):
                    scipy.copyto(self.dPrxy['eta'][i], self.Prxy / (self.eta[i]
                            - int(i == w)), where=CODON_NT_MUT[w])
                _fill_diagonals(self.dPrxy['eta'][i], self._diag_indices)