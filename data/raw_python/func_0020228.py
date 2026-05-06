def values(self):
        '''Iteratir over values of :class:`PairMixin`.'''
        if self.cache.cache is None:
            backend = self.read_backend
            return backend.execute(backend.structure(self).values(),
                                   self.load_values)
        else:
            return self.cache.cache.values()