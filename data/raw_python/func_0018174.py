def _event_funcs(self, event: str) -> Iterable[Callable]:
        """ Returns an Iterable of the functions subscribed to a event.

        :param event: Name of the event.
        :type event: str

        :return: A iterable to do things with.
        :rtype: Iterable
        """
        for func in self._events[event]:
            yield func