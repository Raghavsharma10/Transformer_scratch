def assertSignalNotFired(self, signal, *args, **kwargs):
        """
        Assert that a signal was fired with appropriate arguments.

        :param signal:
            The :class:`Signal` that should not have been fired.
            Typically this is ``SomeClass.on_some_signal`` reference
        :param args:
            List of positional arguments passed to the signal handler
        :param kwargs:
            List of keyword arguments passed to the signal handler
        """
        event = (signal, args, kwargs)
        self.assertNotIn(
            event, self._events_seen,
            "\nSignal unexpectedly fired: {}\n".format(event))