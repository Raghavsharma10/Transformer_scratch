def set_geometry(self, geom):
        """A convenience function to set the geometry variables.

        Args:
            geom: A tuple containing (thet0, thet, phi0, phi, alpha, beta).
            See the Scatterer class documentation for a description of these
            angles.
        """
        (self.thet0, self.thet, self.phi0, self.phi, self.alpha, 
            self.beta) = geom