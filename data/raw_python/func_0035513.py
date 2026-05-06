def _update_dPrxy(self):
        """Update `dPrxy`."""
        super(ExpCM_fitprefs, self)._update_dPrxy()

        if 'zeta' in self.freeparams:
            tildeFrxyQxy = self.tildeFrxy * self.Qxy
            j = 0
            zetaxterm = scipy.ndarray((self.nsites, N_CODON, N_CODON), dtype='float')
            zetayterm = scipy.ndarray((self.nsites, N_CODON, N_CODON), dtype='float')
            for r in range(self.nsites):
                for i in range(N_AA - 1):
                    zetari = self.zeta[j]
                    zetaxterm.fill(0)
                    zetayterm.fill(0)
                    zetaxterm[r][self._aa_for_x > i] = -1.0 / zetari
                    zetaxterm[r][self._aa_for_x == i] = -1.0 / (zetari - 1.0)
                    zetayterm[r][self._aa_for_y > i] = 1.0 / zetari
                    zetayterm[r][self._aa_for_y == i] = 1.0 / (zetari - 1.0)
                    self.dPrxy['zeta'][j] = tildeFrxyQxy * (zetayterm + zetaxterm)
                    _fill_diagonals(self.dPrxy['zeta'][j], self._diag_indices)
                    j += 1