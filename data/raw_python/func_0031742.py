def search_streams(self, query, hls=False, limit=25, offset=0):
        """Search for streams and return them

        :param query: the query string
        :type query: :class:`str`
        :param hls: If true, only return streams that have hls stream
        :type hls: :class:`bool`
        :param limit: maximum number of results
        :type limit: :class:`int`
        :param offset: offset for pagination
        :type offset: :class:`int`
        :returns: A list of streams
        :rtype: :class:`list` of :class:`models.Stream` instances
        :raises: None
        """
        r = self.kraken_request('GET', 'search/streams',
                                params={'query': query,
                                        'hls': hls,
                                        'limit': limit,
                                        'offset': offset})
        return models.Stream.wrap_search(r)