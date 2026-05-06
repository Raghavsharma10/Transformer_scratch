def push_front(self, value):
        '''Appends a copy of ``value`` to the beginning of the list.'''
        self.cache.push_front(self.value_pickler.dumps(value))