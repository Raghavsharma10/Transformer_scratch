def push_back(self, value):
        '''Appends a copy of *value* at the end of the :class:`Sequence`.'''
        self.cache.push_back(self.value_pickler.dumps(value))
        return self