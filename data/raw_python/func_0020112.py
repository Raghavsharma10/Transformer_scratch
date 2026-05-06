def pop(self, instance):
        '''Remove ``instance`` from the :class:`SessionModel`. Instance
could be a :class:`Model` or an id.

:parameter instance: a :class:`Model` or an ``id``.
:rtype: the :class:`Model` removed from session or ``None`` if
    it was not in the session.
'''
        if isinstance(instance, self.model):
            iid = instance.get_state().iid
        else:
            iid = instance
        instance = None
        for d in (self._new, self._modified, self._deleted):
            if iid in d:
                inst = d.pop(iid)
                if instance is None:
                    instance = inst
                elif inst is not instance:
                    raise ValueError('Critical error: %s is duplicated' % iid)
        return instance