def add_event(self, func: Callable, event: str) -> None:
        """ Adds a function to a event.

        :param func: The function to call when event is emitted
        :type func: Callable

        :param event: Name of the event.
        :type event: str
        """
        self._events[event].add(func)