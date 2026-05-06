def _broadcast_transport_message(self, origin, message):
        """
        Broadcasts an event originating from a transport that does not represent a message from the Pebble.

        :param origin: The type of transport responsible for the message.
        :type origin: .MessageTarget
        :param message: The message from the transport
        """
        self.event_handler.broadcast_event((_EventType.Transport, type(origin), type(message)), message)