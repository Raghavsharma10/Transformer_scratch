def iter_hooks(self, number=-1, etag=None):
        """Iterate over hooks registered on this repository.

        :param int number: (optional), number of hoks to return. Default: -1
            returns all hooks
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :returns: generator of :class:`Hook <github3.repos.hook.Hook>`\ s
        """
        url = self._build_url('hooks', base_url=self._api)
        return self._iter(int(number), url, Hook, etag=etag)