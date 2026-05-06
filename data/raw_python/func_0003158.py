def _set_scatter_signature(self):
        """Mark the amplitude and scattering matrices as up to date.
        """
        self._scatter_signature = (self.thet0, self.thet, self.phi0, self.phi,
            self.alpha, self.beta, self.orient)