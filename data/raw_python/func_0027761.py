def average(self):
        """
        Return the average value (as defined by the AVG implementation in the
        database) of the values specified by this query.

        Note: for non-numeric column types the result of this method will be
        nonsensical.

        @return: a L{float} representing the 'average' value of this column.
        """
        rslt = self._runQuery('SELECT', 'AVG(%s)' % (self._queryTarget,)) or [(0,)]
        assert len(rslt) == 1, 'more than one result: %r' % (rslt,)
        return rslt[0][0]