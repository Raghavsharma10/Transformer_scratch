def sum(self):
        """
        Return the sum of all the values returned by this query.  If no results
        are specified, return None.

        Note: for non-numeric column types the result of this method will be
        nonsensical.

        @return: a number or None.
        """
        res = self._runQuery('SELECT', 'SUM(%s)' % (self._queryTarget,)) or [(0,)]
        assert len(res) == 1, "more than one result: %r" % (res,)
        dbval = res[0][0] or 0
        return self.attribute.outfilter(dbval, _FakeItemForFilter(self.store))