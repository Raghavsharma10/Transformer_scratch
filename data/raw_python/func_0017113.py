def iter_notifications(self, all=False, participating=False, since=None,
                           number=-1, etag=None):
        """Iterates over the notifications for this repository.

        :param bool all: (optional), show all notifications, including ones
            marked as read
        :param bool participating: (optional), show only the notifications the
            user is participating in directly
        :param since: (optional), filters out any notifications updated
            before the given time. This can be a `datetime` or an `ISO8601`
            formatted date string, e.g., 2012-05-20T23:10:27Z
        :type since: datetime or string
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :returns: generator of :class:`Thread <github3.notifications.Thread>`
        """
        url = self._build_url('notifications', base_url=self._api)
        params = {
            'all': all,
            'participating': participating,
            'since': timestamp_parameter(since)
        }
        for (k, v) in list(params.items()):
            if not v:
                del params[k]
        return self._iter(int(number), url, Thread, params, etag)