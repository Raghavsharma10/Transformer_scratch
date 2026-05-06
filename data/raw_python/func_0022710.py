def connect(self, fun):
        """ Connect a function to an event

        The name of the function
        should be on_X, with X the name of the event (e.g. 'on_draw').

        This method is typically used as a decorator on a function
        definition for an event handler.

        Parameters
        ----------
        fun : callable
            The function.
        """
        # Get and check name
        name = fun.__name__
        if not name.startswith('on_'):
            raise ValueError('When connecting a function based on its name, '
                             'the name should start with "on_"')
        eventname = name[3:]
        # Get emitter
        try:
            emitter = self.events[eventname]
        except KeyError:
            raise ValueError(
                'Event "%s" not available on this canvas.' %
                eventname)
        # Connect
        emitter.connect(fun)