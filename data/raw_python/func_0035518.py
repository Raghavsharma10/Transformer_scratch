def _update_dPrxy(self):
        """Update `dPrxy`, accounting for dependence of `phi` on `beta`."""
        super(ExpCM_empirical_phi, self)._update_dPrxy()
        if 'beta' in self.freeparams:
            self.dQxy_dbeta = scipy.zeros((N_CODON, N_CODON), dtype='float')
            for w in range(N_NT):
                scipy.copyto(self.dQxy_dbeta, self.dphi_dbeta[w],
                        where=CODON_NT_MUT[w])
            self.dQxy_dbeta[CODON_TRANSITION] *= self.kappa
            self.dPrxy['beta'] += self.Frxy * self.dQxy_dbeta
            _fill_diagonals(self.dPrxy['beta'], self._diag_indices)