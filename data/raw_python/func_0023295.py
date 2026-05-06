def from_image(cls, header, max_norder, mask=None):
        """
        Creates a `~mocpy.moc.MOC` from an image stored as a FITS file.

        Parameters
        ----------
        header : `astropy.io.fits.Header`
            FITS header containing all the info of where the image is located (position, size, etc...)
        max_norder : int
            The moc resolution.
        mask : `numpy.ndarray`, optional
            A boolean array of the same size of the image where pixels having the value 1 are part of
            the final MOC and pixels having the value 0 are not.

        Returns
        -------
        moc : `~mocpy.moc.MOC`
            The resulting MOC.
        """
        # load the image data
        height = header['NAXIS2']
        width = header['NAXIS1']

        # use wcs from astropy to locate the image in the world coordinates
        w = wcs.WCS(header)

        if mask is not None:
            # We have an array of pixels that are part of of survey
            y, x = np.where(mask)
            pix_crd = np.dstack((x, y))[0]
        else:
            # If we do not have a mask array we create the moc of all the image
            #
            step_pix = 1
            """
            Coords returned by wcs_pix2world method correspond to pixel centers. We want to retrieve the moc pix
            crossing the borders of the image so we have to add 1/2 to the pixels coords before computing the lonlat.
            
            The step between two pix_crd is set to `step_pix` but can be diminished to have a better precision at the 
            borders so that all the image is covered (a too big step does not retrieve all
            the moc pix crossing the borders of the image).
            """
            x, y = np.mgrid[0.5:(width + 0.5 + step_pix):step_pix, 0.5:(height + 0.5 + step_pix):step_pix]
            pix_crd = np.dstack((x.ravel(), y.ravel()))[0]

        frame = wcs.utils.wcs_to_celestial_frame(w)
        world_pix_crd = SkyCoord(w.wcs_pix2world(pix_crd, 1), unit='deg', frame=frame)

        hp = HEALPix(nside=(1 << max_norder), order='nested', frame=ICRS())
        ipix = hp.skycoord_to_healpix(world_pix_crd)
        # remove doubles
        ipix = np.unique(ipix)

        shift = 2 * (AbstractMOC.HPY_MAX_NORDER - max_norder)
        intervals_arr = np.vstack((ipix << shift, (ipix + 1) << shift)).T

        # This MOC will be consistent when one will do operations on the moc (union, inter, ...) or
        # simply write it to a fits or json file
        interval_set = IntervalSet(intervals_arr)

        return cls(interval_set=interval_set)