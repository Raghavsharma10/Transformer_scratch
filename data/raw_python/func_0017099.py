def iter_code_frequency(self, number=-1, etag=None):
        """Iterate over the code frequency per week.

        Returns a weekly aggregate of the number of additions and deletions
        pushed to this repository.

        :param int number: (optional), number of weeks to return. Default: -1
            returns all weeks
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :returns: generator of lists ``[seconds_from_epoch, additions,
            deletions]``

        .. note:: All statistics methods may return a 202. On those occasions,
                  you will not receive any objects. You should store your
                  iterator and check the new ``last_status`` attribute. If it
                  is a 202 you should wait before re-requesting.

        .. versionadded:: 0.7

        """
        url = self._build_url('stats', 'code_frequency', base_url=self._api)
        return self._iter(int(number), url, list, etag=etag)