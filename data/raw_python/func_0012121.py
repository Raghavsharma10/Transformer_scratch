def finish(self, value):
        '''Give the future it's value and trigger any associated callbacks

        :param value: the new value for the future
        :raises:
            :class:`AlreadyComplete <junction.errors.AlreadyComplete>` if
            already complete
        '''
        if self._done.is_set():
            raise errors.AlreadyComplete()

        self._value = value

        for cb in self._cbacks:
            backend.schedule(cb, args=(value,))
        self._cbacks = None

        for wait in list(self._waits):
            wait.finish(self)
        self._waits = None

        for child in self._children:
            child = child()
            if child is None:
                continue
            child._incoming(self, value)
        self._children = None

        self._done.set()