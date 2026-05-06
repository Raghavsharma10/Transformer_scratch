def iter_issue_comments(self, number=-1, etag=None):
        """Iterate over the issue comments on this pull request.

        :param int number: (optional), number of comments to return. Default:
            -1 returns all available comments.
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :returns: generator of :class:`IssueComment <IssueComment>`\ s
        """
        url = self._build_url(base_url=self.links['comments'])
        return self._iter(int(number), url, IssueComment, etag=etag)