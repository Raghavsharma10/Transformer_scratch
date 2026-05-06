def size(self):
        '''Number of elements in the :class:`Structure`.'''
        if self.cache.cache is None:
            return self.read_backend_structure().size()
        else:
            return len(self.cache.cache)