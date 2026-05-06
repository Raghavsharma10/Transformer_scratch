def get_beamarea_pix(self, ra, dec):
        """
        Calculate the beam area in square pixels.

        Parameters
        ----------
        ra, dec : float
            The sky coordinates at which the calculation is made
        dec

        Returns
        -------
        area : float
            The beam area in square pixels.
        """
        parea = abs(self.pixscale[0] * self.pixscale[1])  # in deg**2 at reference coords
        barea = self.get_beamarea_deg2(ra, dec)
        return barea / parea