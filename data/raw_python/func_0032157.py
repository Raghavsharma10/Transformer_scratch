def isOnSilicon(self, ra_deg, dec_deg, padding_pix=DEFAULT_PADDING):
        """Returns True if the given location is observable with a science CCD.

        Parameters
        ----------
        ra_deg : float
            Right Ascension (J2000) in decimal degrees.

        dec_deg : float
            Declination (J2000) in decimal degrees.

        padding : float
            Objects <=  this many pixels off the edge of a channel are counted
            as inside.  This allows one to compensate for the slight
            inaccuracy in `K2fov` that results from e.g. the lack of optical
            distortion modeling.
        """
        ch, col, row = self.getChannelColRow(ra_deg, dec_deg)
        # Modules 3 and 7 are no longer operational
        if ch in self.brokenChannels:
            return False
        # K2fov encodes the Fine Guidance Sensors (FGS) as
        # "channel" numbers 85-88; they are not science CCDs.
        if ch > 84:
            return False
        return self.colRowIsOnSciencePixel(col, row, padding_pix)