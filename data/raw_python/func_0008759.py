def from_file(cls, filename, beam=None):
        """
        Create a new WCSHelper class from a given fits file.

        Parameters
        ----------
        filename : string
            The file to be read

        beam : :class:`AegeanTools.fits_image.Beam` or None
            The synthesized beam. If the supplied beam is None then one is constructed form the header.

        Returns
        -------
        obj : :class:`AegeanTools.wcs_helpers.WCSHelper`
            A helper object
        """
        header = fits.getheader(filename)
        return cls.from_header(header, beam)