def emit(self, event: str, *args, **kwargs) -> None:
        """ Emit an event and run the subscribed functions.

        :param event: Name of the event.
        :type event: str

        .. notes:
            Passing in threads=True as a kwarg allows to run emitted events
            as separate threads. This can significantly speed up code execution
            depending on the code being executed.
        """
        threads = kwargs.pop('threads', None)

        if threads:

            events = [
                Thread(target=f, args=args, kwargs=kwargs) for f in
                self._event_funcs(event)
            ]

            for event in events:
                event.start()

        else:
            for func in self._event_funcs(event):
                func(*args, **kwargs)