def _update_dprx(self):
        """Update `dprx`."""
        if 'beta' in self.freeparams:
            for r in range(self.nsites):
                self.dprx['beta'][r] = self.prx[r] * (self.ln_pi_codon[r]
                        - scipy.dot(self.ln_pi_codon[r], self.prx[r]))
        if 'eta' in self.freeparams:
            boolterm = scipy.ndarray(N_CODON, dtype='float')
            with scipy.errstate(divide='raise', under='raise', over='raise',
                    invalid='raise'):
                for i in range(N_NT - 1):
                    boolterm.fill(0)
                    for j in range(3):
                        boolterm += ((i <= CODON_NT_INDEX[j]).astype('float') /
                                (self.eta[i] - (i == CODON_NT_INDEX[j]).astype(
                                'float')))
                    for r in range(self.nsites):
                        self.dprx['eta'][i][r] = self.prx[r] * (boolterm -
                                scipy.dot(boolterm, self.prx[r]) / self.prx[r].sum())