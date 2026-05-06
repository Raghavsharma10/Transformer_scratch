def _eta_from_phi(self):
        """Update `eta` using current `phi`."""
        self.eta = scipy.ndarray(N_NT - 1, dtype='float')
        etaprod = 1.0
        for w in range(N_NT - 1):
            self.eta[w] = 1.0 - self.phi[w] / etaprod
            etaprod *= self.eta[w]
        _checkParam('eta', self.eta, self.PARAMLIMITS, self.PARAMTYPES)