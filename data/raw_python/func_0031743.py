def followed_streams(self, limit=25, offset=0):
        """Return the streams the current user follows.

        Needs authorization ``user_read``.

        :param limit: maximum number of results
        :type limit: :class:`int`
        :param offset: offset for pagination
        :type offset: :class:`int`
        :returns: A list of streams
        :rtype: :class:`list`of :class:`models.Stream` instances
        :raises: :class:`exceptions.NotAuthorizedError`
        """
        r = self.kraken_request('GET', 'streams/followed',
                                params={'limit': limit,
                                        'offset': offset})
        return models.Stream.wrap_search(r)