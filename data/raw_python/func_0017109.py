def iter_issue_events(self, number=-1, etag=None):
        """Iterates over issue events on this repository.

        :param int number: (optional), number of events to return. Default: -1
            returns all available events
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :returns: generator of
            :class:`IssueEvent <github3.issues.event.IssueEvent>`\ s
        """
        url = self._build_url('issues', 'events', base_url=self._api)
        return self._iter(int(number), url, IssueEvent, etag=etag)