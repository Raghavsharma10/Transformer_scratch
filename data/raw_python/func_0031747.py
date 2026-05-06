def get_channel_access_token(self, channel):
        """Return the token and sig for the given channel

        :param channel: the channel or channel name to get the access token for
        :type channel: :class:`channel` | :class:`str`
        :returns: The token and sig for the given channel
        :rtype: (:class:`unicode`, :class:`unicode`)
        :raises: None
        """
        if isinstance(channel, models.Channel):
            channel = channel.name
        r = self.oldapi_request(
            'GET', 'channels/%s/access_token' % channel).json()
        return r['token'], r['sig']