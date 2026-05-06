def _update_dprx(self):
        """Update `dprx`, accounting for dependence of `phi` on `beta`."""
        super(ExpCM_empirical_phi, self)._update_dprx()
        if 'beta' in self.freeparams:
            dphi_over_phi = scipy.zeros(N_CODON, dtype='float')
            for j in range(3):
                dphi_over_phi += (self.dphi_dbeta / self.phi)[CODON_NT_INDEX[j]]
            for r in range(self.nsites):
                self.dprx['beta'][r] += self.prx[r] * (dphi_over_phi
                        - scipy.dot(dphi_over_phi, self.prx[r]))