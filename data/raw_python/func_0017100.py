def iter_comments_on_commit(self, sha, number=1, etag=None):
        """Iterate over comments for a single commit.

        :param sha: (required), sha of the commit to list comments on
        :type sha: str
        :param int number: (optional), number of comments to return. Default:
            -1 returns all comments
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :returns: generator of
            :class:`RepoComment <github3.repos.comment.RepoComment>`\ s
        """
        url = self._build_url('commits', sha, 'comments', base_url=self._api)
        return self._iter(int(number), url, RepoComment, etag=etag)