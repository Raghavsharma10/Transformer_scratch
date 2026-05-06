def _update_phi(self):
        """Update `phi` using current `eta`."""
        etaprod = 1.0
        for w in range(N_NT - 1):
            self.phi[w] = etaprod * (1 - self.eta[w])
            etaprod *= self.eta[w]
        self.phi[N_NT - 1] = etaprod