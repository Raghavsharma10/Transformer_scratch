def get_beamarea_pix(self, ra, dec):
        """
        Calculate the area of the beam in square pixels.

        Parameters
        ----------
        ra, dec : float
            The sky position (degrees).

        Returns
        -------
        area : float
            The area of the beam in square pixels.
        """
        beam = self.get_pixbeam(ra, dec)
        if beam is None:
            return 0
        return beam.a * beam.b * np.pi