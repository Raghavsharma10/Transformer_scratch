def read_transport_message(self, origin, message_type, timeout=15):
        """
        Blocking read of a transport message that does not indicate a message from the Pebble.
        Will block until a message is received, or it times out.

        .. warning::
           Avoid calling this method from an endpoint callback; doing so is likely to lead to deadlock.

        :param origin: The type of :class:`.MessageTarget` that triggers the message.
        :param message_type: The class of the message to read from the transport.
        :param timeout: The maximum time to wait before raising :exc:`.TimeoutError`.
        :return: The object read from the transport; of the same type as passed to ``message_type``.
        """
        return self.event_handler.wait_for_event((_EventType.Transport, origin, message_type), timeout=timeout)