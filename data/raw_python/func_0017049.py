def iter_events(self, public=False, number=-1, etag=None):
        """Iterate over events performed by this user.

        :param bool public: (optional), only list public events for the
            authenticated user
        :param int number: (optional), number of events to return. Default: -1
            returns all available events.
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :returns: list of :class:`Event <github3.events.Event>`\ s
        """
        path = ['events']
        if public:
            path.append('public')
        url = self._build_url(*path, base_url=self._api)
        return self._iter(int(number), url, Event, etag=etag)