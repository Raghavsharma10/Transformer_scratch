def difference_update(self, values):
        '''Remove an iterable of *values* from the set.'''
        d = self.value_pickler.dumps
        return self.cache.remove(tuple((d(v) for v in values)))