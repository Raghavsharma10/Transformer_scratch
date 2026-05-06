def iter_notifications(self, all=False, participating=False, number=-1,
                           etag=None):
        """Iterate over the user's notification.

        :param bool all: (optional), iterate over all notifications
        :param bool participating: (optional), only iterate over notifications
            in which the user is participating
        :param int number: (optional), how many notifications to return
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :returns: generator of
            :class:`Thread <github3.notifications.Thread>`
        """
        params = None
        if all:
            params = {'all': all}
        elif participating:
            params = {'participating': participating}

        url = self._build_url('notifications')
        return self._iter(int(number), url, Thread, params, etag=etag)