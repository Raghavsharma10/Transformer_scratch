def get_channel(self, name):
        """Return the channel for the given name

        :param name: the channel name
        :type name: :class:`str`
        :returns: the model instance
        :rtype: :class:`models.Channel`
        :raises: None
        """
        r = self.kraken_request('GET', 'channels/' + name)
        return models.Channel.wrap_get_channel(r)