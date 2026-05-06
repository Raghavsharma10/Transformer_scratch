def from_fits(cls, filename):
        """
        Loads a MOC from a FITS file.

        The specified FITS file must store the MOC (i.e. the list of HEALPix cells it contains) in a binary HDU table.

        Parameters
        ----------
        filename : str
            The path to the FITS file.

        Returns
        -------
        result : `~mocpy.moc.MOC` or `~mocpy.tmoc.TimeMOC`
            The resulting MOC.
        """
        table = Table.read(filename)

        intervals = np.vstack((table['UNIQ'], table['UNIQ']+1)).T

        nuniq_interval_set = IntervalSet(intervals)
        interval_set = IntervalSet.from_nuniq_interval_set(nuniq_interval_set)
        return cls(interval_set)