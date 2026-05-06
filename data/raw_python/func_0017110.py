def iter_languages(self, number=-1, etag=None):
        """Iterate over the programming languages used in the repository.

        :param int number: (optional), number of languages to return. Default:
            -1 returns all used languages
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :returns: generator of tuples
        """
        url = self._build_url('languages', base_url=self._api)
        return self._iter(int(number), url, tuple, etag=etag)