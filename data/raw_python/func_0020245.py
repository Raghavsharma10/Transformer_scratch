def rank(self, value):
        '''The rank of a given *value*. This is the position of *value*
in the :class:`OrderedMixin` container.'''
        value = self.value_pickler.dumps(value)
        return self.backend_structure().rank(value)