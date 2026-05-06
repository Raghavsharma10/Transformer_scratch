def add(self, value):
        '''Add *value* to the set'''
        return self.cache.update((self.value_pickler.dumps(value),))