def without(self, other):
        """
        Subtract another Region by performing a difference operation on their pixlists.

        Requires both regions to have the same maxdepth.

        Parameters
        ----------
        other : :class:`AegeanTools.regions.Region`
            The region to be combined.
        """
        # work only on the lowest level
        # TODO: Allow this to be done for regions with different depths.
        if not (self.maxdepth == other.maxdepth): raise AssertionError("Regions must have the same maxdepth")
        self._demote_all()
        opd = set(other.get_demoted())
        self.pixeldict[self.maxdepth].difference_update(opd)
        self._renorm()
        return