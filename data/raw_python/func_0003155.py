def get_geometry(self):
        """A convenience function to get the geometry variables.

        Returns:
            A tuple containing (thet0, thet, phi0, phi, alpha, beta).
            See the Scatterer class documentation for a description of these
            angles.
        """
        return (self.thet0, self.thet, self.phi0, self.phi, self.alpha, 
            self.beta)