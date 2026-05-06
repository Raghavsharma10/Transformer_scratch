def abort(self, klass, exc, tb=None):
        '''Finish this future (maybe early) in an error state

        Takes a standard exception triple as arguments (like returned by
        ``sys.exc_info``) and will re-raise them as the value.

        Any :class:`Dependents` that are children of this one will also be
        aborted.

        :param class klass: the class of the exception
        :param Exception exc: the exception instance itself
        :param traceback tb: the traceback associated with the exception

        :raises:
            :class:`AlreadyComplete <junction.errors.AlreadyComplete>` if
            already complete
        '''
        if self._done.is_set():
            raise errors.AlreadyComplete()

        self._failure = (klass, exc, tb)

        for eb in self._errbacks:
            backend.schedule(eb, args=(klass, exc, tb))
        self._errbacks = None

        for wait in list(self._waits):
            wait.finish(self)
        self._waits = None

        for child in self._children:
            child = child()
            if child is None:
                continue
            child.abort(klass, exc, tb)
        self._children = None

        self._done.set()