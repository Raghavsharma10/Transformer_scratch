def assertSignalFired(self, signal, *args, **kwargs):
        """
        Assert that a signal was fired with appropriate arguments.

        :param signal:
            The :class:`Signal` that should have been fired.
            Typically this is ``SomeClass.on_some_signal`` reference
        :param args:
            List of positional arguments passed to the signal handler
        :param kwargs:
            List of keyword arguments passed to the signal handler
        :returns:
            A 3-tuple (signal, args, kwargs) that describes that event
        """
        event = (signal, args, kwargs)
        self.assertIn(
            event, self._events_seen,
            "\nSignal unexpectedly not fired: {}\n".format(event))
        return event