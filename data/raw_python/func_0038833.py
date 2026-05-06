def mine_items(self, identifiers, params=None, callback=None):
        """Mine metadata from Archive.org items.

        :param identifiers: Archive.org identifiers to be mined.
        :type identifiers: iterable

        :param params: URL parameters to send with each metadata
                       request.
        :type params: dict

        :param callback: A callback function to be called on each
                         :py:class:`aiohttp.client.ClientResponse`.
        :type callback: func
        """
        # By default, don't cache item metadata in redis.
        params = {'dontcache': 1} if not params else {}
        requests = metadata_requests(identifiers, params, callback, self)
        yield from self.mine(requests)