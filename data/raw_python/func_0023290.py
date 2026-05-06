def contains(self, ra, dec, keep_inside=True):
        """
        Returns a boolean mask array of the positions lying inside (or outside) the MOC instance.

        Parameters
        ----------
        ra : `astropy.units.Quantity`
            Right ascension array
        dec : `astropy.units.Quantity`
            Declination array
        keep_inside : bool, optional
            True by default. If so the mask describes coordinates lying inside the MOC. If ``keep_inside``
            is false, contains will return the mask of the coordinates lying outside the MOC.

        Returns
        -------
        array : `~np.ndarray`
            A mask boolean array
        """
        depth = self.max_order
        m = np.zeros(nside2npix(1 << depth), dtype=bool)

        pix_id = self._best_res_pixels()
        m[pix_id] = True

        if not keep_inside:
            m = np.logical_not(m)

        hp = HEALPix(nside=(1 << depth), order='nested')
        pix = hp.lonlat_to_healpix(ra, dec)

        return m[pix]