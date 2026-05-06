def iter_events(self, number=-1, etag=None):
        """Iterate over public events.

        :param int number: (optional), number of events to return. Default: -1
            returns all available events
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :returns: generator of :class:`Event <github3.events.Event>`\ s
        """
        url = self._build_url('events')
        return self._iter(int(number), url, Event, etag=etag)