def union(self, another_moc, *args):
        """
        Union between the MOC instance and other MOCs.

        Parameters
        ----------
        another_moc : `~mocpy.moc.MOC`
            The MOC used for performing the union with self.
        args : `~mocpy.moc.MOC`
            Other additional MOCs to perform the union with.

        Returns
        -------
        result : `~mocpy.moc.MOC`/`~mocpy.tmoc.TimeMOC`
            The resulting MOC.
        """
        interval_set = self._interval_set.union(another_moc._interval_set)
        for moc in args:
            interval_set = interval_set.union(moc._interval_set)

        return self.__class__(interval_set)