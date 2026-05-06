def stats(self, start, end, fields=None):
        '''Perform a multivariate statistic calculation of this
:class:`ColumnTS` from a *start*  date/datetime to an
*end* date/datetime.

:param start: Start date for analysis.
:param end: End date for analysis.
:param fields: Optional subset of :meth:`fields` to perform analysis on.
    If not provided all fields are included in the analysis.
'''
        start = self.pickler.dumps(start)
        end = self.pickler.dumps(end)
        backend = self.read_backend
        return backend.execute(
            backend.structure(self).stats(start, end, fields), self._stats)