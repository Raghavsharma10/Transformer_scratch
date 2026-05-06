def from_lonlat(cls, lon, lat, max_norder):
        """
        Creates a MOC from astropy lon, lat `astropy.units.Quantity`.
        
        Parameters
        ----------
        lon : `astropy.units.Quantity`
            The longitudes of the sky coordinates belonging to the MOC.
        lat : `astropy.units.Quantity`
            The latitudes of the sky coordinates belonging to the MOC.
        max_norder : int
            The depth of the smallest HEALPix cells contained in the MOC.
        
        Returns
        -------
        result : `~mocpy.moc.MOC`
            The resulting MOC
        """
        hp = HEALPix(nside=(1 << max_norder), order='nested')
        ipix = hp.lonlat_to_healpix(lon, lat)

        shift = 2 * (AbstractMOC.HPY_MAX_NORDER - max_norder)
        intervals = np.vstack((ipix << shift, (ipix + 1) << shift)).T

        interval_set = IntervalSet(intervals)
        return cls(interval_set)