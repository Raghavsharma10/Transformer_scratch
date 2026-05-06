def register_transport_endpoint(self, origin, message_type, handler):
        """
        Register a handler for a message received from a transport that does not indicate a message from the connected
        Pebble.

        :param origin: The type of :class:`.MessageTarget` that triggers the message
        :param message_type: The class of the message that is expected.
        :param handler: A callback to be called when a message is received.
        :type handler: callable
        :return: A handle that can be passed to :meth:`unregister_endpoint` to remove the handler.
        """
        return self.event_handler.register_handler((_EventType.Transport, origin, message_type), handler)