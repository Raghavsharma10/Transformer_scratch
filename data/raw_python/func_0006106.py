def _connection_handler(self, event_name, data, channel_name):
        """Handle incoming data.

        :param str event_name: Name of the event.
        :param Any data: Data received.
        :param str channel_name: Name of the channel this event and data belongs to.
        """
        if channel_name in self.channels:
            self.channels[channel_name]._handle_event(event_name, data)