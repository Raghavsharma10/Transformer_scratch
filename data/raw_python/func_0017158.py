def iter_events(self, number=-1):
        """Iterate over events associated with this issue only.

        :param int number: (optional), number of events to return. Default: -1
            returns all events available.
        :returns: generator of
            :class:`IssueEvent <github3.issues.event.IssueEvent>`\ s
        """
        url = self._build_url('events', base_url=self._api)
        return self._iter(int(number), url, IssueEvent)