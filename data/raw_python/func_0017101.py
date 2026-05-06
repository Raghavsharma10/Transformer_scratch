def iter_commit_activity(self, number=-1, etag=None):
        """Iterate over last year of commit activity by week.

        See: http://developer.github.com/v3/repos/statistics/

        :param int number: (optional), number of weeks to return. Default -1
            will return all of the weeks.
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :returns: generator of dictionaries

        .. note:: All statistics methods may return a 202. On those occasions,
                  you will not receive any objects. You should store your
                  iterator and check the new ``last_status`` attribute. If it
                  is a 202 you should wait before re-requesting.

        .. versionadded:: 0.7

        """
        url = self._build_url('stats', 'commit_activity', base_url=self._api)
        return self._iter(int(number), url, dict, etag=etag)