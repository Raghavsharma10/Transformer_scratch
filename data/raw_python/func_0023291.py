def add_neighbours(self):
        """
        Extends the MOC instance so that it includes the HEALPix cells touching its border.

        The depth of the HEALPix cells added at the border is equal to the maximum depth of the MOC instance.

        Returns
        -------
        moc : `~mocpy.moc.MOC`
            self extended by one degree of neighbours.
        """
        # Get the pixels array of the MOC at the its max order.
        ipix = self._best_res_pixels()

        hp = HEALPix(nside=(1 << self.max_order), order='nested')
        # Get the HEALPix array containing the neighbors of ``ipix``.
        # This array "extends" ``ipix`` by one degree of neighbors. 
        extend_ipix = AbstractMOC._neighbour_pixels(hp, ipix)
        
        # Compute the difference between ``extend_ipix`` and ``ipix`` to get only the neighboring pixels
        # located at the border of the MOC.
        neigh_ipix = np.setdiff1d(extend_ipix, ipix)

        shift = 2 * (AbstractMOC.HPY_MAX_NORDER - self.max_order)
        neigh_itv = np.vstack((neigh_ipix << shift, (neigh_ipix + 1) << shift)).T
        # This array of HEALPix neighbors are added to the MOC to get an ``extended`` MOC at its max order.
        self._interval_set = self._interval_set.union(IntervalSet(neigh_itv))
        return self