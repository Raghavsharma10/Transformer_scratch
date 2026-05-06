def remove_event(self, func_name: str, event: str) -> None:
        """ Removes a subscribed function from a specific event.

        :param func_name: The name of the function to be removed.
        :type func_name: str

        :param event: The name of the event.
        :type event: str

        :raise EventDoesntExist if there func_name doesn't exist in event.
        """
        event_funcs_copy = self._events[event].copy()

        for func in self._event_funcs(event):
            if func.__name__ == func_name:
                event_funcs_copy.remove(func)

        if self._events[event] == event_funcs_copy:
            err_msg = "function doesn't exist inside event {} ".format(event)
            raise EventDoesntExist(err_msg)
        else:
            self._events[event] = event_funcs_copy