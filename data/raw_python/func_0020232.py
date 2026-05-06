def irange(self, start=0, end=-1, callback=None, withscores=True,
               **options):
        '''Return the range by rank between start and end.'''
        backend = self.read_backend
        res = backend.structure(self).irange(start, end,
                                             withscores=withscores,
                                             **options)
        if not callback:
            callback = self.load_data if withscores else self.load_values
        return backend.execute(res, callback)