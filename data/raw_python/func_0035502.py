def _update_Frxy(self):
        """Update `Frxy` from `piAx_piAy_beta`, `ln_piAx_piAy_beta`, `omega`, `beta`."""
        self.Frxy.fill(1.0)
        self.Frxy_no_omega.fill(1.0)
        with scipy.errstate(divide='raise', under='raise', over='raise',
                        invalid='ignore'):
            scipy.copyto(self.Frxy_no_omega, -self.ln_piAx_piAy_beta
                    / (1 - self.piAx_piAy_beta), where=scipy.logical_and(
                    CODON_NONSYN, scipy.fabs(1 - self.piAx_piAy_beta) >
                    ALMOST_ZERO))
        scipy.copyto(self.Frxy, self.Frxy_no_omega * self.omega, where=CODON_NONSYN)