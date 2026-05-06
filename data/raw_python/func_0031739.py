def search_channels(self, query, limit=25, offset=0):
        """Search for channels and return them

        :param query: the query string
        :type query: :class:`str`
        :param limit: maximum number of results
        :type limit: :class:`int`
        :param offset: offset for pagination
        :type offset: :class:`int`
        :returns: A list of channels
        :rtype: :class:`list` of :class:`models.Channel` instances
        :raises: None
        """
        r = self.kraken_request('GET', 'search/channels',
                                params={'query': query,
                                        'limit': limit,
                                        'offset': offset})
        return models.Channel.wrap_search(r)