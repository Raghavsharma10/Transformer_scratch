def block_pop_front(self, timeout=10):
        '''Remove the first element from of the list. If no elements are
available, blocks for at least ``timeout`` seconds.'''
        value = yield self.backend_structure().block_pop_front(timeout)
        if value is not None:
            yield self.value_pickler.loads(value)