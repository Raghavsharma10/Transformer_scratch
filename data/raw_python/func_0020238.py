def update(self, values):
        '''Add iterable *values* to the set'''
        d = self.value_pickler.dumps
        return self.cache.update(tuple((d(v) for v in values)))