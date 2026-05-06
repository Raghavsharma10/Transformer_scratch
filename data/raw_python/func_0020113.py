def expunge(self, instance):
        '''Remove *instance* from the :class:`Session`. Instance could be a
:class:`Model` or an id.

:parameter instance: a :class:`Model` or an *id*
:rtype: the :class:`Model` removed from session or ``None`` if
    it was not in the session.
'''
        instance = self.pop(instance)
        instance.session = None
        return instance