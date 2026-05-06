def _update_dPxy(self):
        """Update `dPxy`."""
        if 'kappa' in self.freeparams:
            scipy.copyto(self.dPxy['kappa'], self.Pxy / self.kappa,
                    where=CODON_TRANSITION)
            _fill_diagonals(self.dPxy['kappa'], self._diag_indices)
        if 'omega' in self.freeparams:
            scipy.copyto(self.dPxy['omega'], self.Pxy_no_omega,
                    where=CODON_NONSYN)
            _fill_diagonals(self.dPxy['omega'], self._diag_indices)