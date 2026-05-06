def merge(self, *series, **kwargs):
        '''Merge this :class:`ColumnTS` with several other *series*.

:parameters series: a list of tuples where the nth element is a tuple
    of the form::

    (wight_n, ts_n1, ts_n2, ..., ts_nMn)

The result will be calculated using the formula::

    ts = weight_1*ts_11*ts_12*...*ts_1M1 + weight_2*ts_21*ts_22*...*ts_2M2 +
         ...
'''
        session = self.session
        if not session:
            raise SessionNotAvailable('No session available')
        self.check_router(session.router, *series)
        return self._merge(*series, **kwargs)