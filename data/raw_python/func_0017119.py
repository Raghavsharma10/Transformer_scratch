def iter_tags(self, number=-1, etag=None):
        """Iterates over tags on this repository.

        :param int number: (optional), return up to at most number tags.
            Default: -1 returns all available tags.
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :returns: generator of :class:`RepoTag <github3.repos.tag.RepoTag>`\ s
        """
        url = self._build_url('tags', base_url=self._api)
        return self._iter(int(number), url, RepoTag, etag=etag)