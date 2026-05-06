def _calculate_Phi_x(self):
        """Calculate `Phi_x` (stationary state) from `phi`."""
        self.Phi_x = scipy.ones(N_CODON, dtype='float')
        for codon in range(N_CODON):
            for pos in range(3):
                self.Phi_x[codon] *= self.phi[pos][CODON_NT_INDEX[pos][codon]]
        self.Phi_x /= self.Phi_x.sum()