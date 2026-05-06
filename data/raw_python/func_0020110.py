def add(self, instance, modified=True, persistent=None,
            force_update=False):
        '''Add a new instance to this :class:`SessionModel`.

:param modified: Optional flag indicating if the ``instance`` has been
    modified. By default its value is ``True``.
:param force_update: if ``instance`` is persistent, it forces an update of the
    data rather than a full replacement. This is used by the
    :meth:`insert_update_replace` method.
:rtype: The instance added to the session'''
        if instance._meta.type == 'structure':
            return self._add_structure(instance)
        state = instance.get_state()
        if state.deleted:
            raise ValueError('State is deleted. Cannot add.')
        self.pop(state.iid)
        pers = persistent if persistent is not None else state.persistent
        pkname = instance._meta.pkname()
        if not pers:
            instance._dbdata.pop(pkname, None)  # to make sure it is add action
            state = instance.get_state(iid=None)
        elif persistent:
            instance._dbdata[pkname] = instance.pkvalue()
            state = instance.get_state(iid=instance.pkvalue())
        else:
            action = 'update' if force_update else None
            state = instance.get_state(action=action, iid=state.iid)
        iid = state.iid
        if state.persistent:
            if modified:
                self._modified[iid] = instance
        else:
            self._new[iid] = instance
        return instance