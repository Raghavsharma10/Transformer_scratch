def from_header(cls, header, beam=None, lat=None):
        """
        Create a new WCSHelper class from the given header.

        Parameters
        ----------
        header : `astropy.fits.HDUHeader` or string
            The header to be used to create the WCS helper

        beam : :class:`AegeanTools.fits_image.Beam` or None
            The synthesized beam. If the supplied beam is None then one is constructed form the header.

        lat : float
            The latitude of the telescope.

        Returns
        -------
        obj : :class:`AegeanTools.wcs_helpers.WCSHelper`
            A helper object.
        """
        try:
            wcs = pywcs.WCS(header, naxis=2)
        except:  # TODO: figure out what error is being thrown
            wcs = pywcs.WCS(str(header), naxis=2)

        if beam is None:
            beam = get_beam(header)
        else:
            beam = beam

        if beam is None:
            logging.critical("Cannot determine beam information")

        _, pixscale = get_pixinfo(header)
        refpix = (header['CRPIX1'], header['CRPIX2'])
        return cls(wcs, beam, pixscale, refpix, lat)