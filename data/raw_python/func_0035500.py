def _update_Qxy(self):
        """Update `Qxy` using current `kappa` and `phi`."""
        for w in range(N_NT):
            scipy.copyto(self.Qxy, self.phi[w], where=CODON_NT_MUT[w])
        self.Qxy[CODON_TRANSITION] *= self.kappa