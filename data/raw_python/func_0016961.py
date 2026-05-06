def iter_forks(self, number=-1, etag=None):
        """Iterator of forks of this gist.

        .. versionchanged:: 0.9

            Added params ``number`` and ``etag``.

        :param int number: (optional), number of forks to iterate over.
            Default: -1 will iterate over all forks of this gist.
        :param str etag: (optional), ETag from a previous request to this
            endpoint.
        :returns: generator of :class:`Gist <Gist>`

        """
        url = self._build_url('forks', base_url=self._api)
        return self._iter(int(number), url, Gist, etag=etag)