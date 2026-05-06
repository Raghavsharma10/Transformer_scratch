def from_skycoords(cls, skycoords, max_norder):
        """
        Creates a MOC from an `astropy.coordinates.SkyCoord`.

        Parameters
        ----------
        skycoords : `astropy.coordinates.SkyCoord`
            The sky coordinates that will belong to the MOC.
        max_norder : int
            The depth of the smallest HEALPix cells contained in the MOC.
        
        Returns
        -------
        result : `~mocpy.moc.MOC`
            The resulting MOC
        """
        hp = HEALPix(nside=(1 << max_norder), order='nested')
        ipix = hp.lonlat_to_healpix(skycoords.icrs.ra, skycoords.icrs.dec)

        shift = 2 * (AbstractMOC.HPY_MAX_NORDER - max_norder)
        intervals = np.vstack((ipix << shift, (ipix + 1) << shift)).T

        interval_set = IntervalSet(intervals)
        return cls(interval_set)