def ipop_range(self, start=0, stop=-1, callback=None, withscores=True):
        '''pop a range from the :class:`OrderedMixin`'''
        backend = self.backend
        res = backend.structure(self).ipop_range(start, stop,
                                                 withscores=withscores)
        if not callback:
            callback = self.load_data if withscores else self.load_values
        return backend.execute(res, callback)