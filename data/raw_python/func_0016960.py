def iter_commits(self, number=-1, etag=None):
        """Iter over the commits on this gist.

        These commits will be requested from the API and should be the same as
        what is in ``Gist.history``.

        .. versionadded:: 0.6

        .. versionchanged:: 0.9

            Added param ``etag``.

        :param int number: (optional), number of commits to iterate over.
            Default: -1 will iterate over all commits associated with this
            gist.
        :param str etag: (optional), ETag from a previous request to this
            endpoint.
        :returns: generator of
            :class:`GistHistory <github3.gists.history.GistHistory>`

        """
        url = self._build_url('commits', base_url=self._api)
        return self._iter(int(number), url, GistHistory)