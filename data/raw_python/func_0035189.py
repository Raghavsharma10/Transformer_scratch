def connect(self, listener, pass_signal=False):
        """
        Connect a new listener to this signal

        :param listener:
            The listener (callable) to add
        :param pass_signal:
            An optional argument that controls if the signal object is
            explicitly passed to this listener when it is being fired.
            If enabled, a ``signal=`` keyword argument is passed to the
            listener function.
        :returns:
            None

        The listener will be called whenever :meth:`fire()` or
        :meth:`__call__()` are called.  The listener is appended to the list of
        listeners. Duplicates are not checked and if a listener is added twice
        it gets called twice.
        """
        info = listenerinfo(listener, pass_signal)
        self._listeners.append(info)
        _logger.debug("connect %r to %r", str(listener), self._name)
        # Track listeners in the instances only
        if inspect.ismethod(listener):
            listener_object = listener.__self__
            # Ensure that the instance has __listeners__ property
            if not hasattr(listener_object, "__listeners__"):
                listener_object.__listeners__ = collections.defaultdict(list)
            # Append the signals a listener is connected to
            listener_object.__listeners__[listener].append(self)