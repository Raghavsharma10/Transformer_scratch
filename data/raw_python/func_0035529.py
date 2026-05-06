def _update_Pxy(self):
        """Update `Pxy` using current `omega`, `kappa`, and `Phi_x`."""
        scipy.copyto(self.Pxy_no_omega, self.Phi_x.transpose(),
                where=CODON_SINGLEMUT)
        self.Pxy_no_omega[0][CODON_TRANSITION] *= self.kappa
        self.Pxy = self.Pxy_no_omega.copy()
        self.Pxy[0][CODON_NONSYN] *= self.omega
        _fill_diagonals(self.Pxy, self._diag_indices)