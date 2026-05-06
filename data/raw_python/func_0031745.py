def get_playlist(self, channel):
        """Return the playlist for the given channel

        :param channel: the channel
        :type channel: :class:`models.Channel` | :class:`str`
        :returns: the playlist
        :rtype: :class:`m3u8.M3U8`
        :raises: :class:`requests.HTTPError` if channel is offline.
        """
        if isinstance(channel, models.Channel):
            channel = channel.name

        token, sig = self.get_channel_access_token(channel)
        params = {'token': token, 'sig': sig,
                  'allow_audio_only': True,
                  'allow_source': True}
        r = self.usher_request(
            'GET', 'channel/hls/%s.m3u8' % channel, params=params)
        playlist = m3u8.loads(r.text)
        return playlist