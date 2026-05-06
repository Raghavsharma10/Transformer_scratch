def _update_dPrxy(self):
        """Update `dPrxy`."""
        super(ExpCM_fitprefs, self)._update_dPrxy()

        if 'zeta' in self.freeparams:
            self.dPrxy['zeta'].fill(0.0)
            tildeFrxyQxy = self.tildeFrxy * self.Qxy
            j = 0
            for r in range(self.nsites):
                for i in range(N_AA - 1):
                    zetari = self.zeta[j]
                    self.dPrxy['zeta'][j][r] = tildeFrxyQxy[r] * (
                            ((i == self._aa_for_y).astype('float') -
                            (i == self._aa_for_x).astype('float')) / zetari)
                    j += 1
            for j in range(self.dPrxy['zeta'].shape[0]):
                _fill_diagonals(self.dPrxy['zeta'][j], self._diag_indices)