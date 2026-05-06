def remove_neighbours(self):
        """
        Removes from the MOC instance the HEALPix cells located at its border.

        The depth of the HEALPix cells removed is equal to the maximum depth of the MOC instance.

        Returns
        -------
        moc : `~mocpy.moc.MOC`
            self minus its HEALPix cells located at its border.
        """
        # Get the HEALPix cells of the MOC at its max depth
        ipix = self._best_res_pixels()

        hp = HEALPix(nside=(1 << self.max_order), order='nested')
        # Extend it to include the max depth neighbor cells.
        extend_ipix = AbstractMOC._neighbour_pixels(hp, ipix)

        # Get only the max depth HEALPix cells lying at the border of the MOC
        neigh_ipix = np.setxor1d(extend_ipix, ipix)

        # Remove these pixels from ``ipix``
        border_ipix = AbstractMOC._neighbour_pixels(hp, neigh_ipix)
        reduced_ipix = np.setdiff1d(ipix, border_ipix)

        # Build the reduced MOC, i.e. MOC without its pixels which were located at its border.
        shift = 2 * (AbstractMOC.HPY_MAX_NORDER - self.max_order)
        reduced_itv = np.vstack((reduced_ipix << shift, (reduced_ipix + 1) << shift)).T
        self._interval_set = IntervalSet(reduced_itv)
        return self