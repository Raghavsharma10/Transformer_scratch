def get_SZ(self):
        """Get the S and Z matrices using the current parameters.
        """
        if self.psd_integrator is None:
            (self._S, self._Z) = self.get_SZ_orient()
        else:
            scatter_outdated = self._scatter_signature != (self.thet0, 
                self.thet, self.phi0, self.phi, self.alpha, self.beta, 
                self.orient)            
            psd_outdated = self._psd_signature != (self.psd,)
            outdated = scatter_outdated or psd_outdated

            if outdated:
                (self._S, self._Z) = self.psd_integrator(self.psd, 
                    self.get_geometry())
                self._set_scatter_signature()
                self._set_psd_signature()

        return (self._S, self._Z)