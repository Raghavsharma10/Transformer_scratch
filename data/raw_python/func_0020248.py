def times(self, start, stop, callback=None, **kwargs):
        '''The times between times *start* and *stop*.'''
        s1 = self.pickler.dumps(start)
        s2 = self.pickler.dumps(stop)
        backend = self.read_backend
        res = backend.structure(self).times(s1, s2, **kwargs)
        return backend.execute(res, callback or self.load_keys)