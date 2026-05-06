def iter_releases(self, number=-1, etag=None):
        """Iterates over releases for this repository.

        :param int number: (optional), number of refs to return. Default: -1
            returns all available refs
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :returns: generator of
            :class:`Release <github3.repos.release.Release>`\ s
        """
        url = self._build_url('releases', base_url=self._api)
        iterator = self._iter(int(number), url, Release, etag=etag)
        iterator.headers.update(Release.CUSTOM_HEADERS)
        return iterator