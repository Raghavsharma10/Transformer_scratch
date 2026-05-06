def _event_func_names(self, event: str) -> List[str]:
        """ Returns string name of each function subscribed to an event.

        :param event: Name of the event.
        :type event: str

        :return: Names of functions subscribed to a specific event.
        :rtype: list
        """
        return [func.__name__ for func in self._events[event]]