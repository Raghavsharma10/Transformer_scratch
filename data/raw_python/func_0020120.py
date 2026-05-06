def add(self, instance, modified=True, **params):
        '''Add an ``instance`` to the session.

        If the session is not in a
        :ref:`transactional state <transactional-state>`, this operation
        commits changes to the back-end server immediately.

        :parameter instance: a :class:`Model` instance. It must be registered
            with the :attr:`router` which created this :class:`Session`.
        :parameter modified: a boolean flag indicating if the instance was
            modified.
        :return: the ``instance``.

        If the instance is persistent (it is already stored in the database),
        an updated will be performed, otherwise a new entry will be created
        once the :meth:`commit` method is invoked.
        '''
        sm = self.model(instance)
        instance.session = self
        o = sm.add(instance, modified=modified, **params)
        if modified and not self.transaction:
            transaction = self.begin()
            return transaction.commit(lambda: o)
        else:
            return o