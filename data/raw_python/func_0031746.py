def get_quality_options(self, channel):
        """Get the available quality options for streams of the given channel

        Possible values in the list:

          * source
          * high
          * medium
          * low
          * mobile
          * audio

        :param channel: the channel or channel name
        :type channel: :class:`models.Channel` | :class:`str`
        :returns: list of quality options
        :rtype: :class:`list` of :class:`str`
        :raises: :class:`requests.HTTPError` if channel is offline.
        """
        optionmap = {'chunked': 'source',
                     'high': 'high',
                     'medium': 'medium',
                     'low': 'low',
                     'mobile': 'mobile',
                     'audio_only': 'audio'}
        p = self.get_playlist(channel)
        options = []
        for pl in p.playlists:
            q = pl.media[0].group_id
            options.append(optionmap[q])
        return options