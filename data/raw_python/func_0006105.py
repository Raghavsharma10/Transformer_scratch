def unsubscribe(self, channel_name):
        """Unsubscribe from a channel

        :param str channel_name: The name of the channel to unsubscribe from.
        """
        if channel_name in self.channels:
            self.connection.send_event(
                'pusher:unsubscribe', {
                    'channel': channel_name,
                }
            )
            del self.channels[channel_name]