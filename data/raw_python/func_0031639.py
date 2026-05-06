def register_endpoint(self, endpoint, handler):
        """
        Register a handler for a message received from the Pebble.

        :param endpoint: The type of :class:`.PebblePacket` that is being listened for.
        :type endpoint: .PacketType
        :param handler: A callback to be called when a message is received.
        :type handler: callable
        :return: A handle that can be passed to :meth:`unregister_endpoint` to remove the handler.
        """
        return self.event_handler.register_handler((_EventType.Watch, endpoint), handler)