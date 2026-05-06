def _update_pi_vars(self):
        """Update variables that depend on `pi`.

        These are `pi_codon`, `ln_pi_codon`, `piAx_piAy`, `piAx_piAy_beta`,
        `ln_piAx_piAy_beta`.

        Update using current `pi` and `beta`."""
        with scipy.errstate(divide='raise', under='raise', over='raise',
                invalid='raise'):
            for r in range(self.nsites):
                self.pi_codon[r] = self.pi[r][CODON_TO_AA]
                pim = scipy.tile(self.pi_codon[r], (N_CODON, 1)) # [x][y] is piAy
                self.piAx_piAy[r] = pim.transpose() / pim
            self.ln_pi_codon = scipy.log(self.pi_codon)
            self.piAx_piAy_beta = self.piAx_piAy**self.beta
            self.ln_piAx_piAy_beta = scipy.log(self.piAx_piAy_beta)