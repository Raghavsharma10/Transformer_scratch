def iter_comments(self, number=-1, etag=None):
        """Iterate over the comments on this pull request.

        :param int number: (optional), number of comments to return. Default:
            -1 returns all available comments.
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :returns: generator of :class:`ReviewComment <ReviewComment>`\ s
        """
        url = self._build_url('comments', base_url=self._api)
        return self._iter(int(number), url, ReviewComment, etag=etag)