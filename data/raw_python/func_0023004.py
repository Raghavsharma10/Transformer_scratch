def from_cells(cls, cells):
        """
        Creates a MOC from a numpy array representing the HEALPix cells.

        Parameters
        ----------
        cells : `numpy.ndarray`
            Must be a numpy structured array (See https://docs.scipy.org/doc/numpy-1.15.0/user/basics.rec.html).
            The structure of a cell contains 3 attributes:

            - A `ipix` value being a np.uint64
            - A `depth` value being a np.uint32
            - A `fully_covered` flag bit stored in a np.uint8

        Returns
        -------
        moc : `~mocpy.moc.MOC`
            The MOC.
        """
        shift = (AbstractMOC.HPY_MAX_NORDER - cells["depth"]) << 1

        p1 = cells["ipix"]
        p2 = cells["ipix"] + 1

        intervals = np.vstack((p1 << shift, p2 << shift)).T

        return cls(IntervalSet(intervals))