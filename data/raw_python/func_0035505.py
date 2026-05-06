def _update_prx(self):
        """Update `prx` from `phi`, `pi_codon`, and `beta`."""
        qx = scipy.ones(N_CODON, dtype='float')
        for j in range(3):
            for w in range(N_NT):
                qx[CODON_NT[j][w]] *= self.phi[w]
        frx = self.pi_codon**self.beta
        self.prx = frx * qx
        with scipy.errstate(divide='raise', under='raise', over='raise',
                invalid='raise'):
            for r in range(self.nsites):
                self.prx[r] /= self.prx[r].sum()