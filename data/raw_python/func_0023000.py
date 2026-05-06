def intersection(self, another_moc, *args):
        """
        Intersection between the MOC instance and other MOCs.

        Parameters
        ----------
        another_moc : `~mocpy.moc.MOC`
            The MOC used for performing the intersection with self.
        args : `~mocpy.moc.MOC`
            Other additional MOCs to perform the intersection with.

        Returns
        -------
        result : `~mocpy.moc.MOC`/`~mocpy.tmoc.TimeMOC`
            The resulting MOC.
        """
        interval_set = self._interval_set.intersection(another_moc._interval_set)
        for moc in args:
            interval_set = interval_set.intersection(moc._interval_set)

        return self.__class__(interval_set)