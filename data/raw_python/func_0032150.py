def getOrigin(self, cartesian=False):
        """Return the ra/decs of the channel corners if the S/C
        is pointed at the origin (ra,dec = 0,0)

        Inputs:
        cartesian   (bool) If True, return each channel corner
                    as a unit vector

        Returns:
        A 2d numpy array. Each row represents a channel corner
        The columns are module, output, channel, ra, dec

        If cartestian is True, ra, and dec are replaced by the
        coordinates of a 3 vector
        """
        out = self.origin.copy()

        if cartesian is False:
            out = self.getRaDecs(out)
        return out