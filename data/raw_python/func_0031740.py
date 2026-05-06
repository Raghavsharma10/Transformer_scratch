def get_stream(self, channel):
        """Return the stream of the given channel

        :param channel: the channel that is broadcasting.
                        Either name or models.Channel instance
        :type channel: :class:`str` | :class:`models.Channel`
        :returns: the stream or None, if the channel is offline
        :rtype: :class:`models.Stream` | None
        :raises: None
        """
        if isinstance(channel, models.Channel):
            channel = channel.name

        r = self.kraken_request('GET', 'streams/' + channel)
        return models.Stream.wrap_get_stream(r)