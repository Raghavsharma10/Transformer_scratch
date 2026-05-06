def _update_phi(self):
        """Compute `phi`, `dphi_dbeta`, and `eta` from `g` and `frxy`."""
        self.phi = self._compute_empirical_phi(self.beta)
        _checkParam('phi', self.phi, self.PARAMLIMITS, self.PARAMTYPES)
        self._eta_from_phi()
        dbeta = 1.0e-3
        self.dphi_dbeta = scipy.misc.derivative(self._compute_empirical_phi,
                self.beta, dx=dbeta, n=1, order=5)
        dphi_dbeta_halfdx = scipy.misc.derivative(self._compute_empirical_phi,
                self.beta, dx=dbeta / 2, n=1, order=5)
        assert scipy.allclose(self.dphi_dbeta, dphi_dbeta_halfdx, atol=1e-5,
                rtol=1e-4), ("The numerical derivative dphi_dbeta differs "
                "considerably in value for step dbeta = {0} and a step "
                "half that size, giving values of {1} and {2}.").format(
                dbeta, self.dphi_dbeta, dphi_dbeta_halfdx)