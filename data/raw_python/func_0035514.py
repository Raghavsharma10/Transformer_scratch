def _update_dprx(self):
        """Update `dprx`."""
        super(ExpCM_fitprefs, self)._update_dprx()
        j = 0
        if 'zeta' in self.freeparams:
            self.dprx['zeta'].fill(0)
            for r in range(self.nsites):
                for i in range(N_AA - 1):
                    zetari = self.zeta[j]
                    for a in range(i, N_AA):
                        delta_aAx = (CODON_TO_AA == a).astype('float')
                        self.dprx['zeta'][j][r] += (delta_aAx - (delta_aAx
                                * self.prx[r]).sum())/ (zetari - int(i == a))
                    self.dprx['zeta'][j] *= self.prx[r]
                    j += 1