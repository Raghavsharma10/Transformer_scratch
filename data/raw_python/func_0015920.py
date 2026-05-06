def get_channel(self, name):
        """
        Get a channel by name. To get the names, use get_channels.

        :param string name: Name of channel to get
        :returns dict conn: A channel attribute dictionary.

        """
        name = quote(name, '')
        path = Client.urls['channels_by_name'] % name
        chan = self._call(path, 'GET')
        return chan