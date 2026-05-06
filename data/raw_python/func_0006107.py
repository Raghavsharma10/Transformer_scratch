def _reconnect_handler(self):
        """Handle a reconnect."""
        for channel_name, channel in self.channels.items():
            data = {'channel': channel_name}

            if channel.auth:
                data['auth'] = channel.auth

            self.connection.send_event('pusher:subscribe', data)