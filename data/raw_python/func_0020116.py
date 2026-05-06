def dirty(self):
        '''The set of instances in this :class:`Session` which have
been modified.'''
        return frozenset(chain(*tuple((sm.dirty for sm
                                       in itervalues(self._models)))))