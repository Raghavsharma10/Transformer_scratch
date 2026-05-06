def imulti_stats(self, start=0, end=-1, series=None, fields=None,
                     stats=None):
        '''Perform cross multivariate statistics calculation of
this :class:`ColumnTS` and other optional *series* from *start*
to *end*.

:parameter start: the start rank.
:parameter start: the end rank
:parameter field: name of field to perform multivariate statistics.
:parameter series: a list of two elements tuple containing the id of the
    a :class:`columnTS` and a field name.
:parameter stats: list of statistics to evaluate.
    Default: ['covariance']
'''
        stats = stats or self.default_multi_stats
        backend = self.read_backend
        return backend.execute(
            backend.structure(self).imulti_stats(start, end, fields, series,
                                                 stats), self._stats)