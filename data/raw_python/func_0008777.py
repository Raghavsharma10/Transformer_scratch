def get_beamarea_deg2(self, ra, dec):

        """
        Calculate the area of the beam in square degrees.

        Parameters
        ----------
        ra, dec : float
            The sky position (degrees).

        Returns
        -------
        area : float
            The area of the beam in square degrees.
        """
        beam = self.get_beam(ra, dec)
        if beam is None:
            return 0
        return beam.a * beam.b * np.pi