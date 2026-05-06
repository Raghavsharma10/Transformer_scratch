def subscribe(self, channel_name, auth=None):
        """Subscribe to a channel.

        :param str channel_name: The name of the channel to subscribe to.
        :param str auth: The token to use if authenticated externally.
        :rtype: pysher.Channel
        """
        data = {'channel': channel_name}
        if auth is None:
            if channel_name.startswith('presence-'):
                data['auth'] = self._generate_presence_token(channel_name)
                data['channel_data'] = json.dumps(self.user_data)
            elif channel_name.startswith('private-'):
                data['auth'] = self._generate_auth_token(channel_name)
        else:
            data['auth'] = auth

        self.connection.send_event('pusher:subscribe', data)

        self.channels[channel_name] = Channel(channel_name, self.connection)

        return self.channels[channel_name]