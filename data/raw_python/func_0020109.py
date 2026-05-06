def iterdirty(self):
        '''Ordered iterator over dirty elements.'''
        return iter(chain(itervalues(self._new), itervalues(self._modified)))