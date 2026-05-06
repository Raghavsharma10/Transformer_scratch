def itimes(self, start=0, stop=-1, callback=None, **kwargs):
        '''The times between rank *start* and *stop*.'''
        backend = self.read_backend
        res = backend.structure(self).itimes(start, stop, **kwargs)
        return backend.execute(res, callback or self.load_keys)