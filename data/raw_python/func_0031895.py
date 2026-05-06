def wait_for_event(self, event, timeout=10):
        """
        Block waiting for the given event. Returns the event params.

        :param event: The event to handle.
        :return: The event params.
        :param timeout: The maximum time to wait before raising :exc:`.TimeoutError`.
        """
        return self.__handler.wait_for_event(event, timeout=timeout)