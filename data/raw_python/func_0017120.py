def mark_notifications(self, last_read=''):
        """Mark all notifications in this repository as read.

        :param str last_read: (optional), Describes the last point that
            notifications were checked. Anything updated since this time will
            not be updated. Default: Now. Expected in ISO 8601 format:
            ``YYYY-MM-DDTHH:MM:SSZ``. Example: "2012-10-09T23:39:01Z".
        :returns: bool
        """
        url = self._build_url('notifications', base_url=self._api)
        mark = {'read': True}
        if last_read:
            mark['last_read_at'] = last_read
        return self._boolean(self._put(url, data=dumps(mark)),
                             205, 404)