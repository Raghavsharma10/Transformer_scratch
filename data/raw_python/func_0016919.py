def iter_files(self, number=-1, etag=None):
        """Iterate over the files associated with this pull request.

        :param int number: (optional), number of files to return. Default:
            -1 returns all available files.
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :returns: generator of :class:`PullFile <PullFile>`\ s
        """
        url = self._build_url('files', base_url=self._api)
        return self._iter(int(number), url, PullFile, etag=etag)