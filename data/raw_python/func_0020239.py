def discard(self, value):
        '''Remove an element *value* from a set if it is a member.'''
        return self.cache.remove((self.value_pickler.dumps(value),))