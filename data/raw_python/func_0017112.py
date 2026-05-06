def iter_network_events(self, number=-1, etag=None):
        """Iterates over events on a network of repositories.

        :param int number: (optional), number of events to return. Default: -1
            returns all available events
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :returns: generator of :class:`Event <github3.events.Event>`\ s
        """
        base = self._api.replace('repos', 'networks', 1)
        url = self._build_url('events', base_url=base)
        return self._iter(int(number), url, Event, etag)