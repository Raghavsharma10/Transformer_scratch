def expunge(self, instance=None):
        '''Remove ``instance`` from this :class:`Session`. If ``instance``
is not given, it removes all instances from this :class:`Session`.'''
        if instance is not None:
            sm = self._models.get(instance._meta)
            if sm:
                return sm.expunge(instance)
        else:
            self._models.clear()