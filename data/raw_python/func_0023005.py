def from_json(cls, json_moc):
        """
        Creates a MOC from a dictionary of HEALPix cell arrays indexed by their depth.

        Parameters
        ----------
        json_moc : dict(str : [int]
            A dictionary of HEALPix cell arrays indexed by their depth.

        Returns
        -------
        moc : `~mocpy.moc.MOC` or `~mocpy.tmoc.TimeMOC`
            the MOC.
        """
        intervals = np.array([])
        for order, pix_l in json_moc.items():
            if len(pix_l) == 0:
                continue
            pix = np.array(pix_l)
            p1 = pix
            p2 = pix + 1
            shift = 2 * (AbstractMOC.HPY_MAX_NORDER - int(order))

            itv = np.vstack((p1 << shift, p2 << shift)).T
            if intervals.size == 0:
                intervals = itv
            else:
                intervals = np.vstack((intervals, itv))

        return cls(IntervalSet(intervals))