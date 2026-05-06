def istats(self, start=0, end=-1, fields=None):
        '''Perform a multivariate statistic calculation of this
:class:`ColumnTS` from *start* to *end*.

:param start: Optional index (rank) where to start the analysis.
:param end: Optional index (rank) where to end the analysis.
:param fields: Optional subset of :meth:`fields` to perform analysis on.
    If not provided all fields are included in the analysis.
'''
        backend = self.read_backend
        return backend.execute(
            backend.structure(self).istats(start, end, fields), self._stats)