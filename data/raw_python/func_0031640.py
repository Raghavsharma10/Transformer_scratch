def read_from_endpoint(self, endpoint, timeout=15):
        """
        Blocking read from an endpoint. Will block until a message is received, or it times out. Also see
        :meth:`get_endpoint_queue` if you are considering calling this in a loop.

        .. warning::
           Avoid calling this method from an endpoint callback; doing so is likely to lead to deadlock.

        .. note::
           If you're reading a response to a message you just sent, :meth:`send_and_read` might be more appropriate.

        :param endpoint: The endpoint to read from.
        :type endpoint: .PacketType
        :param timeout: The maximum time to wait before raising :exc:`.TimeoutError`.
        :return: The message read from the endpoint; of the same type as passed to ``endpoint``.
        """
        return self.event_handler.wait_for_event((_EventType.Watch, endpoint), timeout=timeout)