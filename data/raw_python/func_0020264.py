def backend_query(self, **kwargs):
        '''Build and return the :class:`stdnet.utils.async.BackendQuery`.
This is a lazy method in the sense that it is evaluated once only and its
result stored for future retrieval.'''
        q = self.construct()
        return q if isinstance(q, EmptyQuery) else q.backend_query(**kwargs)