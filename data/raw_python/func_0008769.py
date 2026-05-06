def get_beamarea_deg2(self, ra, dec):
        """
        Calculate the area of the synthesized beam in square degrees.

        Parameters
        ----------
        ra, dec : float
            The sky coordinates at which the calculation is made.

        Returns
        -------
        area : float
            The beam area in square degrees.
        """
        barea = abs(self.beam.a * self.beam.b * np.pi)  # in deg**2 at reference coords
        if self.lat is not None:
            barea /= np.cos(np.radians(dec - self.lat))
        return barea